import os
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import UNet_PyTorch_Data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Step 4: Run a Quick Overfit Test (Sanity Check)
# Before training on your entire dataset of hundreds of slices, 
# take just 1 single slice and train your model on it for 50–100 epochs.

# Why? If the model can successfully overfit and memorize a single slice, your data loading pipeline, 
# model architecture, and loss function are bug-free. 
# If it fails to overfit a single image, there is an indexing or dimension mismatch in your tensors.

class DoubleConv(nn.Module):
    """Hjälpklass: (Konvolucion => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SinoNet(nn.Module):
    """
    Sino-Net: U-Net baserat nätverk för sinogram-dekomposition.
    - Input: 2 kanaler (High-energy & Low-energy sinograms)
    - Output: 3 kanaler (Material-sinograms: Adipose, Fibroglandular, Calcification)
    - Struktur: 4 ner- och uppsamplingssteg (encoder-decoder) med skip connections.
    """
    def __init__(self, in_channels=2, out_channels=3, features=[16, 32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1. Encoder (Ner-samplingsdel med 4 nivåer)
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Bottleneck (Mitten / Djupaste nivån)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 2. Decoder (Upp-samplingsdel med 4 nivåer)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Slutligt 1x1 konvolutionslager för att mappa till 3 material-kanaler
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder-fas
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = reversed(list(skip_connections))

        # Decoder-fas med skip connections
        for idx, level in enumerate(range(0, len(self.ups), 2)):
            x = self.ups[level](x)
            skip_connection = next(skip_connections)

            # Säkerställ att dimensionerna matchar exakt (vid udda bildstorlekar)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            # Sätt ihop med skip-kopplingen
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[level + 1](concat_x)

        out = self.final_conv(x)
        return F.relu(out)


class SinoDataset(Dataset):
    """
    Dataset för SinoNet: 
    - Input: Hög- och lågenergi-sinogram (2 kanaler)
    - Target: Material-sinogram [Adipose, Fibroglandular, Calcification] (3 kanaler)
    """
    def __init__(self, data_dir, sino_dir, slice_indices):
        self.data_dir = data_dir
        self.sino_dir = sino_dir
        self.slice_indices = slice_indices
        
        self.path_high = os.path.join(data_dir, "highkVpTransmission.npy")
        self.path_low = os.path.join(data_dir, "lowkVpTransmission.npy")
        self.path_adip = os.path.join(sino_dir, "Sinogram_Adipose.npy")
        self.path_gland = os.path.join(sino_dir, "Sinogram_Fibroglandular.npy")
        self.path_calc = os.path.join(sino_dir, "Sinogram_Calcification.npy")
        
        # --- VIKTIGT: Spara mmap-objekten med 'self.' så __getitem__ kommer åt dem ---
        self.mmap_hi = np.load(self.path_high, mmap_mode="r")
        self.mmap_lo = np.load(self.path_low, mmap_mode="r")
        self.mmap_ad = np.load(self.path_adip, mmap_mode="r")
        self.mmap_gl = np.load(self.path_gland, mmap_mode="r")
        self.mmap_ca = np.load(self.path_calc, mmap_mode="r")
        
        print(f"[INFO] High transmission shape: {self.mmap_hi.shape}")

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        s_idx = self.slice_indices[idx]
        
        hi = self.mmap_hi[s_idx].astype(np.float32)
        lo = self.mmap_lo[s_idx].astype(np.float32)
        ad = self.mmap_ad[s_idx].astype(np.float32)
        gl = self.mmap_gl[s_idx].astype(np.float32)
        ca = self.mmap_ca[s_idx].astype(np.float32)
        
        inputs = np.stack([hi, lo], axis=0)        # Shape: (2, 256, 1024)
        targets = np.stack([ad, gl, ca], axis=0)   # Shape: (3, 512, 1024)
        
        # Konvertera till tensorer
        inputs_t = torch.tensor(inputs)
        targets_t = torch.tensor(targets)

        # --- NORMALISERING ---
        # 1. Normalisera targets till [0, 1] baserat på ungefärliga maxvärden för kanalerna
        # Kanal 0 (Adipose): max ~ 15.0
        # Kanal 1 (Fibroglandular): max ~ 10.0
        # Kanal 2 (Calcification): max ~ 0.2
        max_vals = torch.tensor([15.0, 10.0, 0.2], dtype=torch.float32).view(3, 1, 1)
        targets_t = targets_t / max_vals  # Nu ligger targets mellan 0 och 1 (ungefär)

        # (Valfritt) Om du även vill normalisera inputs, t.ex. dela med ett känt maxvärde för transmission:
        # inputs_t = inputs_t / 1000.0 
        # ---------------------
        
        # Om targets har höjd 512 men inputs har 256, skala ner targets till 256
        if targets_t.shape[1] != inputs_t.shape[1]:
            # F.interpolate vill ha 4D-tensor [Batch, Kanaler, Höjd, Bredd], så vi lägger till en temporär batch-dimension
            targets_t = torch.nn.functional.interpolate(
                targets_t.unsqueeze(0), 
                size=(inputs_t.shape[1], inputs_t.shape[2]), 
                mode="bilinear", 
                align_corners=False
            ).squeeze(0) # Ta bort batch-dimensionen igen
            
        return inputs_t, targets_t


def train_sinonet(epochs=50, batch_size=1, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Tränar på enhet: {device}")

    # Skapa modell och skicka till GPU
    model = SinoNet(in_channels=2, out_channels=3).to(device)
    
    # Exempel: Använd slices 0 till 1000 för träning
    train_slices = list(range(0, 10))
    
    dataset = SinoDataset(data_dir="/home/nidrazuk/LPD_MMDNet_data/introduction",sino_dir="/home/nidrazuk/LPD_MMDNet_data/model", slice_indices=train_slices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Nollställ gradienter
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Beräkna loss
            loss = criterion(outputs, targets)

            # Backward pass och optimering
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

    # Spara tränad modell
    torch.save(model.state_dict(), "sinonet_baseline.pth")
    print("[INFO] Träning klar! Modell sparad som 'sinonet_baseline.pth'.")

# Snabb test-körning för att verifiera att dimensionerna stämmer
if __name__ == "__main__":
    model = SinoNet(in_channels=2, out_channels=3)
    
    # Ändra till sinogrammets riktiga dimensioner (256 x 1024)
    dummy_input = torch.randn(1, 2, 256, 1024)
    output = model(dummy_input)
    
    print("Input shape: ", dummy_input.shape)
    print("Output shape:", output.shape)  # Skall vara [1, 3, 256, 1024]

    # Kör träningen
    train_sinonet(epochs=50, batch_size=2)
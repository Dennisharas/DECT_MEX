import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from UNet_test import SinoNet, SinoDataset, DoubleConv  # Importerar dina klasser från träningsfilen

def evaluate_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Kör utvärdering på enhet: {device}")

    # 1. Ladda modellen
    model = SinoNet(in_channels=2, out_channels=3, features=[16, 32, 64, 128]).to(device)
    model.load_state_dict(torch.load("sinonet_baseline.pth", map_location=device,weights_only=False))
    model.eval()

    # 2. Skapa dataset för en test-slice (t.ex. slice nr 0)
    data_dir = "/home/nidrazuk/LPD_MMDNet_data/introduction"
    sino_dir = "/home/nidrazuk/LPD_MMDNet_data/model"
    test_dataset = SinoDataset(data_dir, sino_dir, slice_indices=[0])
    
    inputs, targets = test_dataset[0]  # Hämtar första test-slicen (256x1024)
    
    # Skicka till enhet och lägg till batch-dimension [1, Channels, Height, Width]
    inputs_dev = inputs.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(inputs_dev)
        
    # Flytta tillbaka till CPU och ta bort batch-dimensionen
    pred = outputs.cpu().squeeze(0).numpy()       # Shape: (3, 256, 1024)
    target = targets.numpy()                     # Shape: (3, 256, 1024)
    inp = inputs.numpy()                         # Shape: (2, 256, 1024)

    # # Skala tillbaka prediktionerna till ursprungsskala med samma maxvärden
    # max_vals = np.array([15.0, 10.0, 0.2], dtype=np.float32)[:, None, None]
    # pred_original = pred * max_vals

    # 3. Beräkna enkla mätvärden (t.ex. MSE per materialkanal à la Cullblom)
    materials = ["Adipose", "Fibroglandular", "Calcification"]
    print("\n--- Utvärderingsresultat (MSE per kanal) ---")
    for i, mat_name in enumerate(materials):
        mse = np.mean((pred[i] - target[i]) ** 2)
        print(f"{mat_name} MSE: {mse:.6f}")

    # 4. Visualisera med Matplotlib
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    # Rad 1: Inputs (High & Low energy transmission)
    axes[0, 0].imshow(inp[0], cmap='gray')
    axes[0, 0].set_title("High-kVp Transmission")
    axes[0, 1].imshow(inp[1], cmap='gray')
    axes[0, 1].set_title("Low-kVp Transmission")
    axes[0, 2].axis('off') # Tom ruta för layoutens skull

    # Rad 2 & 3: Predicted vs Ground Truth för de 3 materialen
    for i, mat_name in enumerate(materials):
        # Prediction
        im1 = axes[1, i].imshow(pred[i], cmap='viridis')
        axes[1, i].set_title(f"Pred: {mat_name}")
        fig.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

        # Ground Truth (Facit)
        im2 = axes[2, i].imshow(target[i], cmap='viridis')
        axes[2, i].set_title(f"Target: {mat_name}")
        fig.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("evaluation_result.png", dpi=300)
    print("\n[INFO] Visualisering sparad som 'evaluation_result.png'!")
    plt.show()

if __name__ == "__main__":
    evaluate_and_visualize()
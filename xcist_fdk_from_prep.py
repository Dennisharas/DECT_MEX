import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import gecatsim as xc
from gecatsim.pyfiles import CommonTools as CT

# ----- hitta fdk_equiAngle.py och lägg rätt mapp i sys.path -----
HERE = os.path.dirname(__file__)

# kandidat 1: ~/Desktop/DECT_Breast_Project/main/gecatsim/reconstruction/pyfiles
RECON_PATH_1 = os.path.abspath(os.path.join(
    HERE, "..", "main", "gecatsim", "reconstruction", "pyfiles"
))

# kandidat 2: om projektet råkar ha en annan struktur
RECON_PATH_2 = os.path.abspath(os.path.join(
    HERE, "..", "main", "reconstruction", "pyfiles"
))

print("[DEBUG] RECON_PATH_1 =", RECON_PATH_1)
print("[DEBUG] RECON_PATH_2 =", RECON_PATH_2)

FILENAME = "fdk_equiAngle.py"

chosen_path = None
for p in (RECON_PATH_1, RECON_PATH_2):
    candidate = os.path.join(p, FILENAME)
    if os.path.exists(candidate):
        chosen_path = p
        break

if chosen_path is None:
    raise RuntimeError(
        f"Hittar inte {FILENAME} i:\n  {RECON_PATH_1}\neller\n  {RECON_PATH_2}.\n"
        "Kontrollera i VS Code exakt var filen ligger."
    )

print("[INFO] Använder reconstruction-path:", chosen_path)
sys.path.append(chosen_path)

from fdk_equiAngle import fdk_equiAngle



def main():
    ap = argparse.ArgumentParser(
        description="Kör XCIST FDK (fdk_equiAngle) direkt på en existing .prep."
    )
    ap.add_argument(
        "--prefix",
        required=True,
        help=(
            "Filprefix till din simulering UTAN ändelse. "
            "Ex: ../main/EXE/results/breast2d_60kVp_Cu0.2"
        ),
    )
    args = ap.parse_args()

    # ---------------- 1) Registrera sökvägar som i sim_2d.py ----------------
    HERE = os.path.dirname(__file__)
    PKG = os.path.abspath(os.path.join(HERE, "..", "main", "gecatsim"))

    CT.my_path.paths = {
        "cfg": os.path.join(PKG, "cfg"),
        "spectrum": os.path.join(PKG, "spectrum"),
        "bowtie": os.path.join(PKG, "bowtie"),
        "material": os.path.join(PKG, "material"),
        "phantom": os.path.join(PKG, "phantom"),
        "lib": os.path.join(PKG, "lib"),
    }

    print("[INFO] PKG =", PKG)
    print("[INFO] cfg-path =", CT.my_path.paths["cfg"])

    # ---------------- 2) Skapa CatSim-objekt och hämta cfg ----------------
    ct = xc.CatSim(
        "Phantom_Default",
        "Scanner_Default",
        "Physics_Default",
        "Protocol_Default",
        "Recon_Default",
    )

    # samma voxelphantom som i sim_2d.py
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.filename = os.path.expanduser(
        "~/Desktop/DECT_Breast_Project/xcist_voxelized_bundle_auto2d_calc/UncompressedBreast1_z202/breast_voxelized_2D.json"
    )


    cfg = ct.get_current_cfg()

    # säkerställ rimliga recon-parametrar (kan justeras senare)
    cfg.recon.fov = 320.0              # mm, FOV
    cfg.recon.imageSize = 800          # antal pixlar i X/Y
    cfg.recon.sliceCount = 1           # 2D-recon 
    cfg.recon.sliceThickness = 0.3   # eller vad din voxelstorlek är i z

    #cfg.recon.sliceThickness = (
       # cfg.scanner.detectorRowSize * cfg.scanner.sid / cfg.scanner.sdd
    #)
    cfg.recon.centerOffset = [-15.0, 10.0, 0.0]
    cfg.recon.startAngle = 0.0
    cfg.recon.kernelType = "soft"
    cfg.recon.printReconParameters = True 


    # ---------------- 3) Läs .prep (EXAKT samma format som i plot_sin.py) ---
    prep_path = os.path.abspath(args.prefix + ".prep")
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Hittar inte .prep: {prep_path}")

    V = int(cfg.protocol.viewCount)
    R = int(cfg.scanner.detectorRowCount)
    C = int(cfg.scanner.detectorColCount)
    print(f"[INFO] Förväntad prep-dimension: V={V}, R={R}, C={C}")
    prep = CT.rawread(prep_path, [V, R, C], "float")  # (V,R,C)
    print(f"[INFO] Läs .prep, shape={prep.shape}")

    # ---------------- 4) Kör XCIST FDK direkt ----------------
    print("[INFO] Kör fdk_equiAngle(cfg, prep)...")
    vol = fdk_equiAngle(cfg, prep)  # förväntad shape (X, Y, Z)
    print("[INFO] FDK klar, volym shape:", vol.shape)

    # ---------------- 5) Spara volym + en slice ----------------
    outdir = os.path.join(os.path.dirname(prep_path), "xcist_fdk")
    os.makedirs(outdir, exist_ok=True)

    npy_path = os.path.join(outdir, "recon_fdk60.npy")
    np.save(npy_path, vol)
    print("[INFO] Sparade volym till:", npy_path)

    # ta mitt-slice i Z (eller enda slicen om sliceCount=1)
    zmid = vol.shape[2] // 2
    img = vol[:, :, zmid]

    vmin, vmax = np.percentile(img, [5, 99])

    png_path = os.path.join(outdir, "recon_fdk60_mid.png")
    plt.figure()
    plt.title("XCIST FDK (mitt-slice)")
    plt.imshow(img.T, cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    print("[INFO] Sparade bild till:", png_path)


if __name__ == "__main__":
    main()

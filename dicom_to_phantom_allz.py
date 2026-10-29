import os
import json
import numpy as np
import pydicom
from glob import glob 
from scipy.ndimage import binary_erosion


import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label

# ==============================
# PATHS (SÄTT DESSA)
# ==============================
# ==============================
# PATHS
# ==============================
DICOM_DIR = os.path.expanduser(
    "~/compressed_dicom/compressedBreast1"
)

OUT_ROOT  = os.path.expanduser(
    "~/DECT_Breast_Project/xcist_voxelized_bundle_auto2d_calc/UncompressedBreast1_allz2"
)

os.makedirs(OUT_ROOT, exist_ok=True)

os.makedirs(OUT_ROOT, exist_ok=True)

# ==============================
# SETTINGS
# ==============================
AIR_THR = 0.08          # threshold i 0–1 efter normalisering (justera vid behov)
MORPH_ITERS = 1          # hur aggressivt masken städas (lite räcker ofta)
GLAND_THR = 0.50 
SKIN_THICK_PIX = 3 

# Calcification (”förkalkning”) – liten och slumpmässig men 3D-konsistent
ADD_CALC = True
CALC_SEED = 123
N_CALC_BLOBS = 6        # antal små klumpar
CALC_RADIUS_VOX = 2     # radie i voxlar (2 => väldigt liten)
CALC_TO_MATERIAL = "bone"  # lägg calc i "bone"-fraktionen (kan ändras)

# ==============================
# LOAD DICOM SERIES (ROBUST SORT)
# ==============================
dicom_files = glob(os.path.join(DICOM_DIR, "*.dcm"))
print(f"[INFO] Found {len(dicom_files)} DICOM files in {DICOM_DIR}")
if len(dicom_files) == 0:
    raise RuntimeError(f"No DICOM files found in: {DICOM_DIR}")

def sort_key(path: str):
    ds = pydicom.dcmread(path, stop_before_pixels=True)

    # 1) ImagePositionPatient (vanligast för CT)
    if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
        try:
            return (0, float(ds.ImagePositionPatient[2]))
        except Exception:
            pass

    # 2) SliceLocation (ibland finns denna)
    if hasattr(ds, "SliceLocation") and ds.SliceLocation is not None:
        try:
            return (1, float(ds.SliceLocation))
        except Exception:
            pass

    # 3) InstanceNumber (ofta finns)
    if hasattr(ds, "InstanceNumber") and ds.InstanceNumber is not None:
        try:
            return (2, int(ds.InstanceNumber))
        except Exception:
            pass

    # 4) fallback: filnamn (sist utväg)
    return (3, os.path.basename(path))

files = sorted(dicom_files, key=sort_key)
print(f"[INFO] Loaded {len(files)} DICOM slices (sorted)")

# ==============================
# READ VOLUME
# ==============================
slices = []
for f in files:
    ds = pydicom.dcmread(f)
    img = ds.pixel_array.astype(np.float32)
    slices.append(img)

vol = np.stack(slices, axis=0)  # (Z, Y, X)
Z, Y, X = vol.shape
print(f"[INFO] Volume shape: {vol.shape}")

# ==============================
# NORMALIZE (0–1)
# ==============================
vmin, vmax = float(vol.min()), float(vol.max())
vol = (vol - vmin) / (vmax - vmin + 1e-12)

# ==============================
# BUILD BREAST MASK (3D)
# ==============================
# Grov mask: allt som inte är "luft"
mask = vol > AIR_THR

# Städa masken lite (tar bort små prickar / stänger små hål)
# kör per slice för stabilitet
clean = np.zeros_like(mask, dtype=bool)
for z in range(Z):
    m = mask[z]
    m = binary_opening(m, iterations=MORPH_ITERS)
    m = binary_closing(m, iterations=MORPH_ITERS)
    m = binary_fill_holes(m)
    # behåll största komponenten (tar bort småöar)
    lab, n = label(m)
    if n > 0:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        keep = counts.argmax()
        m = (lab == keep)
    clean[z] = m

breast_mask = clean

# Nollställ allt utanför bröstet -> eliminerar "luft-material"
vol = vol * breast_mask.astype(np.float32)

# ==============================
# CREATE 3D CALCIFICATION MASK (PILLARS THROUGH Z)
# ==============================
calc_mask = np.zeros((Z, Y, X), dtype=bool)

def add_disk_2d(mask2d, cy, cx, r):
    yy, xx = np.ogrid[:Y, :X]
    disk = (yy - cy)**2 + (xx - cx)**2 <= r**2
    mask2d |= disk

if ADD_CALC:
    rng = np.random.default_rng(CALC_SEED)

    # välj punkter som ligger i bröstet för ALLA z (så calc syns i varje slice)
    stable_xy = np.all(breast_mask, axis=0)  # (Y, X)
    coords_xy = np.argwhere(stable_xy)

    # om bröstet ändrar form så att inget är sant i alla z, fallback: välj från mitten-slice
    if coords_xy.shape[0] == 0:
        mid = Z // 2
        coords_xy = np.argwhere(breast_mask[mid])
        print("[WARN] No (x,y) present in all slices; using middle slice for calc placement.")

    if coords_xy.shape[0] == 0:
        print("[WARN] Breast mask is empty. No calcifications added.")
    else:
        centers = coords_xy[rng.integers(0, coords_xy.shape[0], size=N_CALC_BLOBS)]

        for (cy, cx) in centers:
            cy, cx = int(cy), int(cx)
            for z in range(Z):
                if breast_mask[z, cy, cx]:
                    add_disk_2d(calc_mask[z], cy, cx, CALC_RADIUS_VOX)

        calc_mask &= breast_mask

# ==============================
# LOOP OVER Z SLICES
# ==============================
for z in range(Z):
    out_dir = os.path.join(OUT_ROOT, f"z{z:03d}")
    os.makedirs(out_dir, exist_ok=True)

    img = vol[z]  # 0–1 och noll utanför bröstet
    msk = breast_mask[z]
    csk = calc_mask[z]

    # ==============================
    # MATERIAL CLASSIFICATION (DISCRETE + SKIN AS RIM)
    # ==============================
    # Skin = tunn kant runt bröstet
    eroded = binary_erosion(msk, iterations=SKIN_THICK_PIX)
    if not np.any(eroded):
        eroded = binary_erosion(msk, iterations=max(1, SKIN_THICK_PIX - 1))
    skin_mask = msk & (~eroded)

    interior = msk & (~skin_mask)

    # Adip/Gland i interior baserat på intensitet
    vol_adip  = ((img < GLAND_THR) & interior).astype(np.float32)
    vol_gland = ((img >= GLAND_THR) & interior).astype(np.float32)

    # Skin separat
    vol_skin  = skin_mask.astype(np.float32)

    # Bone används bara för calc här
    vol_bone  = np.zeros((Y, X), dtype=np.float32)

 

    # ---- insert calcification consistently ----
    if ADD_CALC and np.any(csk):
        if CALC_TO_MATERIAL == "bone":
            vol_bone[csk] = 1.0
        elif CALC_TO_MATERIAL == "skin":
            vol_skin[csk] = 1.0
        elif CALC_TO_MATERIAL == "gland":
            vol_gland[csk] = 1.0
        else:
            vol_bone[csk] = 1.0  # fallback

    # ---- renormalize fractions inside breast ----
    # gör så att adip+gland+skin+bone ≈ 1 inne i bröstet
    s = vol_adip + vol_gland + vol_skin + vol_bone
    s_safe = np.where(s > 1e-8, s, 1.0)
    vol_adip  = np.where(msk, vol_adip / s_safe, 0.0).astype(np.float32)
    vol_gland = np.where(msk, vol_gland / s_safe, 0.0).astype(np.float32)
    vol_skin  = np.where(msk, vol_skin / s_safe, 0.0).astype(np.float32)
    vol_bone  = np.where(msk, vol_bone / s_safe, 0.0).astype(np.float32)

    # ---- write bins ----
    vol_adip.tofile(os.path.join(out_dir, "breast2d_vol_1.bin"))
    vol_gland.tofile(os.path.join(out_dir, "breast2d_vol_2.bin"))
    vol_skin.tofile(os.path.join(out_dir, "breast2d_vol_3.bin"))
    vol_bone.tofile(os.path.join(out_dir, "breast2d_vol_4.bin"))

    # ---- write json ----
    phantom = {
        "n_materials": 4,
        "mat_name": [
            "ICRU_adipose_adult2",
            "ICRU_breast_adult2",
            "ICRU_skin_adult",
            "ICRU_skeleton_cortical_bone_adult"
        ],
        "mat_density": [0.95, 1.02, 1.09, 1.92],
        "volumefractionmap_filename": [
            "breast2d_vol_1.bin",
            "breast2d_vol_2.bin",
            "breast2d_vol_3.bin",
            "breast2d_vol_4.bin"
        ],
        "volumefractionmap_datatype": ["float", "float", "float", "float"],
        "cols": [X, X, X, X],
        "rows": [Y, Y, Y, Y],
        "slices": [1, 1, 1, 1],
        "x_size": [1.0, 1.0, 1.0, 1.0],
        "y_size": [1.0, 1.0, 1.0, 1.0],
        "z_size": [400.0, 400.0, 400.0, 400.0],
        "x_offset": [X / 2, X / 2, X / 2, X / 2],
        "y_offset": [Y / 2, Y / 2, Y / 2, Y / 2],
        "z_offset": [0.5, 0.5, 0.5, 0.5]
    }

    with open(os.path.join(out_dir, "breast_voxelized_2D.json"), "w") as f:
        json.dump(phantom, f, indent=2)

    # ==============================
    # SAVE PNGs
    # ==============================
    # 1) själva slicen
    plt.figure()
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(out_dir, "slice.png"), dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    # 2) overlay calc
    plt.figure()
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    if ADD_CALC:
        # calc i rött
        overlay = np.zeros((Y, X, 4), dtype=np.float32)
        overlay[..., 0] = csk.astype(np.float32)  # R
        overlay[..., 3] = (csk.astype(np.float32) * 0.9)  # alpha
        plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(out_dir, "overlay_calc.png"), dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    # ==============================
    # 3) MATERIAL OVERLAY (DISCRETE COLORS)
    # ==============================
    rgb = np.zeros((Y, X, 3), dtype=np.float32)

    # Starta med svart bakgrund
    # Lägg på material:
    rgb[vol_adip > 0, 0] = 1.0   # Röd adip
    rgb[vol_gland > 0, 1] = 1.0  # Grön gland
    rgb[vol_skin > 0, 2] = 1.0   # Blå skin

    # Calc = vit (överlagrar allt)
    if ADD_CALC:
        rgb[csk, :] = 1.0

    plt.figure()
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(out_dir, "materials_overlay.png"),
                dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"[OK] Wrote phantom + PNGs for z = {z:03d} -> {out_dir}")

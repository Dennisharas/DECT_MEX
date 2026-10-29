# run_eval_mmdstyle.py
import os, json
import numpy as np
import gzip
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.ndimage import map_coordinates
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import zoom

# (valfritt) om du vill ha tabell som csv:
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
GT_TRANSPOSE = False
SAVE_DEBUG_IMAGES = True          # sparar overlay + dbg-mask (bara baseline)
SAVE_HISTS_SCATTER = False        # sätt True om du vill spara histogram/scatter för baseline

RECON_FOV_MM = 320.0
RECON_IMAGE_SIZE = 512

RECON_CENTER_OFFSET_MM   = (-15.0, 10.0)   # recon.centerOffset (x,y) mm
PHANTOM_CENTER_OFFSET_MM = (0.0, -70.0)    # phantom.centerOffset (x,y) mm

# ===== MANUELL FINJUSTERING (behåll din fungerande) =====
MANUAL_FINE_DY = 0.0   # upp
MANUAL_FINE_DX = 0.0   # vänster

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FDK_DIR = os.path.join(PROJECT_ROOT, "DECT_Breast_Project", "UncompressedBreast1_CT", "z350", "xcist_fdk")
FDK40_NPY = os.path.join(FDK_DIR, "recon_fdk40.npy")
FDK60_NPY = os.path.join(FDK_DIR, "recon_fdk60.npy")

# --- NEW: Point to LPD_MMDNet_data training data ---
LPD_DATA_DIR = os.path.join(PROJECT_ROOT, "LPD_MMDNet_data", "training_data")
SLICE_IDX = 40  # The specific slice index you want to evaluate

OUT_DIR = os.path.join(os.path.dirname(__file__), "eval_outputs_mmdstyle")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# EXPERIMENT LIST (DIN "GRID" AV METODER)
# Anpassa här: lägg till/ta bort mappar/metoder
# ============================================================
EXPERIMENTS = [
    {
        "method": "DeNoise",
        "params": {},  # här kan du lägga t.ex. {"sigma": 2.0} om du vill logga parametrar
        "matdecomp_dir": os.path.join(FDK_DIR, "image_matdecomp_DeNoise"),
    },
    {
        "method": "IterTV",
        "params": {},
        "matdecomp_dir": os.path.join(FDK_DIR, "image_matdecomp_iterative_tv"),
    },
    {
        "method": "DataDriven",
        "params": {},
        "matdecomp_dir": os.path.join(FDK_DIR, "image_matdecomp_datadriven"),
    },
    # Exempel:
    # {"method":"RobustLS","params":{},"matdecomp_dir":os.path.join(FDK_DIR,"image_matdecomp_robust_ls")},
    # {"method":"Calibrated","params":{},"matdecomp_dir":os.path.join(FDK_DIR,"image_matdecomp_calibrated")},
]

# ============================================================
# Helpers
# ============================================================

# Definiera sökvägar till din nya datamapp
LPD_DATA_DIR = "/home/nidrazuk/LPD_MMDNet_data/training_data"

def load_lpd_slice(slice_idx: int):
    """
    Loads a single slice efficiently from compressed .npy.gz files
    without permanently holding the entire 3D stack in memory if possible.
    """
    #data_path = os.path.join(LPD_DATA_DIR, "training_data")
    
    # Helper to load a single slice safely
    def load_slice_from_gz(filename):
        filepath = os.path.join(LPD_DATA_DIR, filename)
        with gzip.open(filepath, "rb") as f:
            # Using mmap or seeking isn't directly supported on gzip streams, 
            # but we can load and immediately slice to minimize lingering overhead.
            arr = np.load(f, allow_pickle=True)
            return arr[slice_idx].astype(np.float32)

    slice_adip = load_slice_from_gz("Phantom_Adipose.npy.gz")
    slice_gland = load_slice_from_gz("Phantom_Fibroglandular.npy.gz")
    slice_calc = load_slice_from_gz("Phantom_Calcification.npy.gz")
    
    return slice_adip, slice_gland, slice_calc

def ensure_3d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x[:, :, None]
    return x

def _to_float32(x): return np.asarray(x, dtype=np.float32)
def ensure_0_1(x): return np.clip(_to_float32(x), 0.0, 1.0)

def load_shape_from_fdk(fdk_npy_path: str):
    vol = np.load(fdk_npy_path, mmap_mode="r")
    return tuple(vol.shape)

def infer_square_2d_shape_from_bin(path: str, dtype=np.float32):
    arr = np.fromfile(path, dtype=dtype)
    n = int(np.sqrt(arr.size))
    if n * n != arr.size:
        raise ValueError(f"Kan inte gissa kvadratisk shape för {path}. arr.size={arr.size}")
    return (n, n, 1)

def read_bin(path: str, shape, dtype=np.float32):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hittar inte: {path}")
    arr = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"Fel storlek: {path} arr.size={arr.size} expected={expected} shape={shape}")
    return arr.reshape(shape)

def maybe_transpose_gt(vol3d, do_transpose: bool):
    return vol3d.transpose(1, 0, 2) if do_transpose else vol3d

def load_gt_meta_from_json(json_path: str):
    with open(json_path, "r") as f:
        meta = json.load(f)
    cols = int(meta["cols"][0])
    rows = int(meta["rows"][0])
    x_size = float(meta["x_size"][0])
    y_size = float(meta["y_size"][0])
    x_off  = float(meta["x_offset"][0])
    y_off  = float(meta["y_offset"][0])
    return {"cols": cols, "rows": rows, "x_size": x_size, "y_size": y_size, "x_offset": x_off, "y_offset": y_off}

def save_gray_png(path, img2d, title="img"):
    plt.figure(figsize=(6, 6))
    plt.imshow(img2d.T, origin="lower", cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_overlay_png(path, gt_mask2d, pred_mask2d, title="overlay"):
    gt = gt_mask2d.astype(bool)
    pr = pred_mask2d.astype(bool)
    rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.float32)
    rgb[..., 1] = gt.astype(np.float32)  # green
    rgb[..., 0] = pr.astype(np.float32)  # red
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb.transpose(1, 0, 2), origin="lower")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def shift2d_subpixel(img2d, dy, dx, order=1):
    return ndi_shift(img2d.astype(np.float32), shift=(dy, dx), order=order,
                     mode="constant", cval=0.0, prefilter=(order>1))

def dice_mask(a, b, eps=1e-8):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.sum(a & b)
    denom = np.sum(a) + np.sum(b)
    return float((2.0 * inter + eps) / (denom + eps))

def estimate_shift_com(ref_mask2d, mov_mask2d):
    ref = ref_mask2d.astype(np.float32)
    mov = mov_mask2d.astype(np.float32)
    if ref.sum() < 10 or mov.sum() < 10:
        return 0.0, 0.0, 999.0
    cy_ref, cx_ref = center_of_mass(ref)
    cy_mov, cx_mov = center_of_mass(mov)
    dy = cy_ref - cy_mov
    dx = cx_ref - cx_mov
    return float(dy), float(dx), 0.0

# ============================================================
# Affine GT->Recon resampling (DIN FUNKTION)
# ============================================================
def resample_gt_to_recon_affine(
    gt2d, gt_meta,
    recon_shape_xy, recon_fov_mm,
    recon_center_offset_mm, phantom_center_offset_mm,
    A, order, pix_shift=(0.0, 0.0)
):
    Nx, Ny = recon_shape_xy
    dx = float(recon_fov_mm) / float(Nx)
    dy = float(recon_fov_mm) / float(Ny)
    cx = (Nx - 1) / 2.0
    cy = (Ny - 1) / 2.0

    uu = np.arange(Nx, dtype=np.float32)
    vv = np.arange(Ny, dtype=np.float32)
    dy_pix, dx_pix = pix_shift

    U, V = np.meshgrid(uu, vv, indexing="ij")
    U = U - dy_pix
    V = V - dx_pix

    tx = (U - cx) * dx
    ty = (V - cy) * dy

    xw = A[0, 0] * tx + A[0, 1] * ty + float(recon_center_offset_mm[0])
    yw = A[1, 0] * tx + A[1, 1] * ty + float(recon_center_offset_mm[1])

    xph = xw - float(phantom_center_offset_mm[0])
    yph = yw - float(phantom_center_offset_mm[1])

    ic = (float(gt_meta["cols"]) - 1.0) / 2.0
    jc = (float(gt_meta["rows"]) - 1.0) / 2.0

    i = xph / float(gt_meta["x_size"]) + ic
    j = yph / float(gt_meta["y_size"]) + jc

    coords = np.vstack([j.ravel(), i.ravel()])  # (row=j, col=i)
    out = map_coordinates(gt2d.astype(np.float32), coords, order=order, mode="constant", cval=0.0)
    return out.reshape((Nx, Ny)).astype(np.float32)

def candidate_A_matrices():
    I    = np.array([[ 1, 0],[ 0, 1]], dtype=np.float32)
    R90  = np.array([[ 0,-1],[ 1, 0]], dtype=np.float32)
    R180 = np.array([[-1, 0],[ 0,-1]], dtype=np.float32)
    R270 = np.array([[ 0, 1],[-1, 0]], dtype=np.float32)
    Mx = np.array([[-1, 0],[ 0, 1]], dtype=np.float32)
    My = np.array([[ 1, 0],[ 0,-1]], dtype=np.float32)
    Md = np.array([[ 0, 1],[ 1, 0]], dtype=np.float32)
    Ma = np.array([[ 0,-1],[-1, 0]], dtype=np.float32)
    return [("I", I), ("R90", R90), ("R180", R180), ("R270", R270),
            ("Mx", Mx), ("My", My), ("Md", Md), ("Ma", Ma)]

# ============================================================
# Metrics
# ============================================================
def rmse_masked(pred, gt, mask):
    m = mask.astype(bool)
    diff = pred[m] - gt[m]
    return float(np.sqrt(np.mean(diff**2))) if np.any(m) else float("nan")

def cnr(sig_vals, bg_vals, eps=1e-12):
    if sig_vals.size == 0 or bg_vals.size == 0:
        return float("nan")
    print(f"Sig adip pixlar: {np.sum(sig_vals)}, Bg adip pixlar: {np.sum(bg_vals)}")
    return float((np.mean(sig_vals) - np.mean(bg_vals)) / (np.sqrt(np.var(sig_vals) + np.var(bg_vals)) + eps))

# ============================================================
# Load GT once
# ============================================================
def load_gt_all(target_shape=(800, 800)):
    """
    Loads ground truth slices directly from LPD_MMDNet_data/training_data.
    """
    adip_path = os.path.join(LPD_DATA_DIR, "Phantom_Adipose.npy")
    gland_path = os.path.join(LPD_DATA_DIR, "Phantom_Fibroglandular.npy")
    calc_path = os.path.join(LPD_DATA_DIR, "Phantom_Calcification.npy")

    gt_ad = np.load(adip_path, mmap_mode="r")[SLICE_IDX].astype(np.float32)
    gt_gl = np.load(gland_path, mmap_mode="r")[SLICE_IDX].astype(np.float32)
    gt_ca = np.load(calc_path, mmap_mode="r")[SLICE_IDX].astype(np.float32)

    # Beräkna zoom-faktor (t.ex. 800 / 512 = 1.5625)
    zoom_y = target_shape[0] / gt_ad.shape[0]
    zoom_x = target_shape[1] / gt_ad.shape[1]

    # Skala om till 800x800 (order=1 för mjukare fraktioner, order=0 för binära/calc)
    gt_ad_resized = zoom(gt_ad, (zoom_y, zoom_x), order=1)
    gt_gl_resized = zoom(gt_gl, (zoom_y, zoom_x), order=1)
    gt_ca_resized = zoom(gt_ca, (zoom_y, zoom_x), order=0)

    # Säkerställ 3D-form (H, W, 1)
    gt_ad_2d = ensure_3d(gt_ad_resized)
    gt_gl_2d = ensure_3d(gt_gl_resized)
    gt_ca_2d = ensure_3d(gt_ca_resized)

    # Dummy meta to keep compatibility with the rest of the script structure
    gt_meta = {"cols": gt_ad_2d.shape[1], "rows": gt_ad_2d.shape[0], "x_size": 1.0, "y_size": 1.0}

    return gt_ad_2d, gt_gl_2d, gt_ca_2d, gt_meta

# ============================================================
# Load prediction for one experiment (NPY eller BIN)
# ============================================================
def load_pred_from_dir(matdecomp_dir: str, recon_shape_xyz):
    # NPY
    npy_adip  = os.path.join(matdecomp_dir, "f_adip_raw.npy")
    npy_gland = os.path.join(matdecomp_dir, "f_gland_raw.npy")
    npy_calc  = os.path.join(matdecomp_dir, "f_calc_raw.npy")

    # BIN
    bin_adip  = os.path.join(matdecomp_dir, "frac_adip.bin")
    bin_gland = os.path.join(matdecomp_dir, "frac_gland.bin")
    bin_calc  = os.path.join(matdecomp_dir, "frac_calc.bin")

    inside_bin = os.path.join(matdecomp_dir, "inside_mask.bin")

    if os.path.exists(npy_adip) and os.path.exists(npy_gland) and os.path.exists(npy_calc):
        pred_adip  = ensure_3d(ensure_0_1(np.load(npy_adip).astype(np.float32)))
        pred_gland = ensure_3d(ensure_0_1(np.load(npy_gland).astype(np.float32)))
        pred_calc  = ensure_3d(ensure_0_1(np.load(npy_calc).astype(np.float32)))
        src = "NPY"
    elif os.path.exists(bin_adip) and os.path.exists(bin_gland) and os.path.exists(bin_calc):
        pred_adip  = ensure_3d(ensure_0_1(read_bin(bin_adip,  recon_shape_xyz, dtype=np.float32)))
        pred_gland = ensure_3d(ensure_0_1(read_bin(bin_gland, recon_shape_xyz, dtype=np.float32)))
        pred_calc  = ensure_3d(ensure_0_1(read_bin(bin_calc,  recon_shape_xyz, dtype=np.float32)))
        src = "BIN"
    else:
        raise FileNotFoundError(
            f"Missing fraction files in: {matdecomp_dir}\n"
            "Expected f_*_raw.npy or frac_*.bin"
        )

    if os.path.exists(inside_bin):
        inside = read_bin(inside_bin, recon_shape_xyz, dtype=np.float32) > 0.5
        inside = ensure_3d(inside)
        inside_src = "inside_mask.bin"
    else:
        # fallback (men OBS: i din “gamla” kod funkar detta för single-run)
        inside = ensure_3d((pred_adip + pred_gland + pred_calc) > 1e-3)
        inside_src = "fallback_sum"

    print(f"[DEBUG] load_pred: {matdecomp_dir} | src={src} | inside_src={inside_src} | inside_mean={float(np.mean(inside.astype(np.float32))):.4f}")
    return pred_adip, pred_gland, pred_calc, inside

# ============================================================
# Pick a stable inside_ref (så alla metoder evalas på samma mask)
# ============================================================
def pick_inside_ref(recon_shape_xyz):
    # 1) välj första experiment som faktiskt har inside_mask.bin
    for exp in EXPERIMENTS:
        p = os.path.join(exp["matdecomp_dir"], "inside_mask.bin")
        if os.path.exists(p):
            inside = read_bin(p, recon_shape_xyz, dtype=np.float32) > 0.5
            inside = ensure_3d(inside)
            m = float(np.mean(inside.astype(np.float32)))
            s = int(np.sum(inside))
            print(f"[INFO] inside_ref from {exp['method']} | mean={m:.4f} sum={s}")
            # skydd mot helbildsmask
            if m < 0.99 and s > 1000:
                return inside

    # 2) annars: fall back till baseline-load (kan funka men riskerar helbild)
    print("[WARN] No inside_mask.bin found in any experiment. Falling back to baseline sum-mask (risk).")
    baseline = EXPERIMENTS[0]
    pred_a, pred_g, pred_c, inside = load_pred_from_dir(baseline["matdecomp_dir"], recon_shape_xyz)
    return inside

def save_material_overlay(slice_idx: int, out_dir: str, lpd_data_dir: str):
    """
    Skapar och sparar en RGB-overlaybild av materialen för den aktuella slicen.
    Färgkodning:
      - Röd = Fett (Adipose)
      - Grön = Körtelvävnad (Fibroglandular)
      - Blå = Kalcifikationer (Calcification)
    """
    adip_path = os.path.join(lpd_data_dir, "Phantom_Adipose.npy")
    gland_path = os.path.join(lpd_data_dir, "Phantom_Fibroglandular.npy")
    calc_path = os.path.join(lpd_data_dir, "Phantom_Calcification.npy")

    # Ladda slice med mmap_mode för att vara minnessnål
    adip = np.load(adip_path, mmap_mode="r")[slice_idx].astype(np.float32)
    gland = np.load(gland_path, mmap_mode="r")[slice_idx].astype(np.float32)
    calc = np.load(calc_path, mmap_mode="r")[slice_idx].astype(np.float32)

    H, W = adip.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Mappa till RGB-kanaler
    rgb[..., 0] = np.clip(adip, 0, 1)        # Röd = Fett
    rgb[..., 1] = np.clip(gland, 0, 1)       # Grön = Körtel
    rgb[..., 2] = np.clip(calc * 5.0, 0, 1)  # Blå = Kalcifikation (förstärkt synlighet)

    out_path = os.path.join(out_dir, f"material_overlay_z{slice_idx}.png")

    # Rita och spara bilden
    plt.figure(figsize=(7, 7))
    plt.imshow(rgb.transpose(1, 0, 2), origin="lower")
    plt.title(f"Material Overlay - Slice {slice_idx}\n(Röd=Fett, Grön=Körtel, Blå=Kalcifikation)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"[INFO] Material-overlay sparad till: {out_path}")

# ============================================================
# MAIN
# ============================================================
def main():

    # rensa output
    for fn in os.listdir(OUT_DIR):
        if fn.endswith(".png") or fn.endswith(".npz") or fn.endswith(".csv") or fn.endswith(".json"):
            try:
                os.remove(os.path.join(OUT_DIR, fn))
            except:
                pass

    # Exempel på hur du kan verifiera att körtelvävnad finns på slicen:
    slice_idx = 150  # Byt till det index du vill testa (t.ex. mitt i volymen)
    adip, gland, calc = load_lpd_slice(slice_idx)

    print(f"Slice {slice_idx} - Glandular max/mean:", np.max(gland), np.mean(gland))
    print(f"Slice {slice_idx} - Adipose max/mean:", np.max(adip), np.mean(adip))

    # recon shape
    shape = load_shape_from_fdk(FDK40_NPY)
    Nx, Ny, Nz = shape
    if Nz != 1:
        raise ValueError("Den här evalen antar 2D (nz=1).")
    recon_shape_xyz = shape
    print("[INFO] Recon shape:", shape)

    # mu images (valfritt)
    mu40 = np.load(FDK40_NPY).astype(np.float32)[:, :, 0]
    mu60 = np.load(FDK60_NPY).astype(np.float32)[:, :, 0]

    # load GT once
    gt_ad, gt_gl, gt_ca, gt_meta = load_gt_all()

    # ============================================================
    # INSIDE REF (gemensam mask för jämförelse)
    # ============================================================
    inside_ref = pick_inside_ref(recon_shape_xyz)
    pred_mask2d = inside_ref[:, :, 0].astype(bool)

    # ============================================================
    # ALIGNMENT (EN GÅNG) baserat på inside_ref
    # ============================================================

    # Since LPD_MMDNet_data is pre-aligned to the data grid:
    best_name = "Identity"
    best_A = np.array([[1, 0], [0, 1]], dtype=np.float32)
    best_dy, best_dx = 0.0, 0.0
    best_dice = 1.0
    
    # Directly assign the loaded slices as gt_adip, gt_gland, gt_calc
    gt_adip = ensure_0_1(gt_ad)
    gt_gland = ensure_0_1(gt_gl)
    gt_calc = ensure_0_1(gt_ca)

    gt_breast = (gt_adip + gt_gland + gt_calc) > 1e-6
    
    # Use inside_ref or fallback directly
    inside_ref = pick_inside_ref(recon_shape_xyz)
    eval_mask = (inside_ref & gt_breast)
    M = eval_mask[:, :, 0].astype(bool)
    print("[INFO] eval_mask pixels:", int(np.sum(M)))    

    run_tag = f"A{best_name}_dy{best_dy:.2f}_dx{best_dx:.2f}_man{MANUAL_FINE_DY}_{MANUAL_FINE_DX}"
    print("[INFO] Best A:", best_name, "dice=", best_dice, "shift(dy,dx)=", (best_dy, best_dx))

    # ============================================================
    # Bygg aligned GT-material (EN GÅNG) med pix_shift
    # ============================================================
    gt_ad_2d = resample_gt_to_recon_affine(gt_ad[:, :, 0], gt_meta, (Nx, Ny), RECON_FOV_MM,
                                          RECON_CENTER_OFFSET_MM, PHANTOM_CENTER_OFFSET_MM,
                                          best_A, order=1, pix_shift=(best_dy, best_dx))
    gt_gl_2d = resample_gt_to_recon_affine(gt_gl[:, :, 0], gt_meta, (Nx, Ny), RECON_FOV_MM,
                                          RECON_CENTER_OFFSET_MM, PHANTOM_CENTER_OFFSET_MM,
                                          best_A, order=1, pix_shift=(best_dy, best_dx))
    gt_ca_2d = resample_gt_to_recon_affine(gt_ca[:, :, 0], gt_meta, (Nx, Ny), RECON_FOV_MM,
                                          RECON_CENTER_OFFSET_MM, PHANTOM_CENTER_OFFSET_MM,
                                          best_A, order=0, pix_shift=(best_dy, best_dx))

    gt_adip  = ensure_0_1(gt_ad_2d)[:, :, None]
    gt_gland = ensure_0_1(gt_gl_2d)[:, :, None]
    gt_calc  = ensure_0_1(gt_ca_2d)[:, :, None]

    gt_breast = (gt_adip + gt_gland + gt_calc) > 1e-6
    # gemensam eval-mask (fair comparison)
    eval_mask = (inside_ref & gt_breast)
    M = eval_mask[:, :, 0].astype(bool)
    print("[INFO] eval_mask pixlar:", int(np.sum(M)))

    # Debug overlay (bara en gång)
    if SAVE_DEBUG_IMAGES:
        save_overlay_png(os.path.join(OUT_DIR, f"overlay_gt_vs_insideRef_{run_tag}.png"),
                         gt_breast[:, :, 0], inside_ref[:, :, 0],
                         title=f"GT vs inside_ref {run_tag}")
        save_gray_png(os.path.join(OUT_DIR, f"dbg_gt_breast_{run_tag}.png"),
                      gt_breast[:, :, 0].astype(np.float32), f"GT breast {run_tag}")
        save_gray_png(os.path.join(OUT_DIR, f"dbg_inside_ref_{run_tag}.png"),
                      inside_ref[:, :, 0].astype(np.float32), f"inside_ref {run_tag}")

    # För CNR: GT-baserade signal/background-regioner (samma som din)
    # gt_adip_bin  = gt_adip[:, :, 0]  > 0.5
    # gt_gland_bin = gt_gland[:, :, 0] > 0.5
    # gt_calc_bin  = gt_calc[:, :, 0]  > 0.1

    gt_adip_bin  = (gt_adip[:, :, 0] > 0.2).astype(bool)
    gt_gland_bin = (gt_gland[:, :, 0] > 0.2).astype(bool)
    gt_calc_bin  = (gt_calc[:, :, 0] > 0.1).astype(bool)

    print(f"[DEBUG MASK] Innan M -> gt_adip: {np.sum(gt_adip_bin)}, gt_gland: {np.sum(gt_gland_bin)}, M: {np.sum(M)}")

    bg_for_adip  = gt_gland_bin
    bg_for_gland = gt_adip_bin
    bg_for_calc  = (gt_adip_bin | gt_gland_bin) & (~gt_calc_bin)

    sig_adip  = gt_adip_bin & M
    sig_gland = gt_gland_bin & M
    sig_calc  = gt_calc_bin & M
    print(f"[DEBUG MASK] Efter M -> sig_adip: {np.sum(sig_adip)}, sig_gland: {np.sum(sig_gland)}")
    bg_adip  = bg_for_adip & M
    bg_gland = bg_for_gland & M
    bg_calc  = bg_for_calc & M

    # ============================================================
    # LOOP ÖVER METODER (GRID) + spara tabell
    # ============================================================
    rows = []
    for exp in EXPERIMENTS:
        try:
            pred_adip, pred_gland, pred_calc, inside_this = load_pred_from_dir(exp["matdecomp_dir"], recon_shape_xyz)

            row = {
                "method": exp["method"],
                "params": json.dumps(exp.get("params", {}), sort_keys=True),
                "matdecomp_dir": exp["matdecomp_dir"],
                "best_A": best_name,
                "best_dy": float(best_dy),
                "best_dx": float(best_dx),
                "n_eval_pix": int(np.sum(M)),

                "rmse_adip":  rmse_masked(pred_adip[:, :, 0],  gt_adip[:, :, 0],  M),
                "rmse_gland": rmse_masked(pred_gland[:, :, 0], gt_gland[:, :, 0], M),
                "rmse_calc":  rmse_masked(pred_calc[:, :, 0],  gt_calc[:, :, 0],  M),

                "cnr_adip":  cnr(pred_adip[:, :, 0][sig_adip],  pred_adip[:, :, 0][bg_adip]),
                "cnr_gland": cnr(pred_gland[:, :, 0][sig_gland], pred_gland[:, :, 0][bg_gland]),
                "cnr_calc":  cnr(pred_calc[:, :, 0][sig_calc],  pred_calc[:, :, 0][bg_calc]),
            }
            rows.append(row)

            print(f"[OK] {exp['method']}: RMSE(adip)={row['rmse_adip']:.4f}  CNR(gland)={row['cnr_gland']:.4f}")

        except Exception as e:
            print(f"[FAIL] {exp['method']} -> {e}")

    # tabell + sortering
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["rmse_mean"] = df[["rmse_adip", "rmse_gland", "rmse_calc"]].mean(axis=1)
        df = df.sort_values(["rmse_mean", "method"]).reset_index(drop=True)

    csv_path = os.path.join(OUT_DIR, "eval_table.csv")
    json_path = os.path.join(OUT_DIR, "eval_table.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print("\n==================== COMPARISON TABLE ====================")
    if len(df) > 0:
        print(df[["method","params","rmse_adip","rmse_gland","rmse_calc","cnr_adip","cnr_gland","cnr_calc","n_eval_pix"]])
    else:
        print("(no successful rows)")
    print("==========================================================\n")
    print("[INFO] Saved:", csv_path)
    print("[INFO] Saved:", json_path)

    # (valfritt) spara baseline-plots/hist/scatter
    if SAVE_HISTS_SCATTER and len(EXPERIMENTS) > 0:
        # baseline = första lyckade raden
        base_method = df.loc[0, "method"] if len(df) > 0 else EXPERIMENTS[0]["method"]
        base_dir = None
        for exp in EXPERIMENTS:
            if exp["method"] == base_method:
                base_dir = exp["matdecomp_dir"]
        if base_dir is not None:
            pred_adip, pred_gland, pred_calc, _ = load_pred_from_dir(base_dir, recon_shape_xyz)

            def save_histogram(path, pred, gt, mask, title):
                m = mask.astype(bool)
                p = pred[m].ravel()
                g = gt[m].ravel()
                if p.size == 0 or g.size == 0:
                    return
                plt.figure(figsize=(6,4))
                plt.hist(g, bins=80, alpha=0.6, density=True, label="GT")
                plt.hist(p, bins=80, alpha=0.6, density=True, label="Pred")
                plt.title(title)
                plt.xlabel("fraction")
                plt.ylabel("density")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path, dpi=200)
                plt.close()

            def save_scatter_mu(path, mu40, mu60, gt_labels, mask, nmax=60000):
                m = mask.astype(bool).ravel()
                if np.sum(m) == 0:
                    return
                idx = np.where(m)[0]
                if idx.size > nmax:
                    rng = np.random.default_rng(0)
                    idx = rng.choice(idx, size=nmax, replace=False)

                x = mu40.ravel()[idx]
                y = mu60.ravel()[idx]
                lab = gt_labels.ravel()[idx]

                plt.figure(figsize=(5.5,5))
                for k, name in [(1,"adip"), (2,"gland"), (3,"calc")]:
                    sel = (lab == k)
                    if np.any(sel):
                        plt.scatter(x[sel], y[sel], s=2, alpha=0.35, label=name)
                plt.xlabel("mu40")
                plt.ylabel("mu60")
                plt.title("Scatter in (mu40, mu60) colored by GT")
                plt.legend(markerscale=4)
                plt.tight_layout()
                plt.savefig(path, dpi=200)
                plt.close()

            save_histogram(os.path.join(OUT_DIR, "hist_adip.png"),
                           pred_adip[:, :, 0], gt_adip[:, :, 0], M,
                           f"Adip fraction: GT vs Pred ({base_method})")
            save_histogram(os.path.join(OUT_DIR, "hist_gland.png"),
                           pred_gland[:, :, 0], gt_gland[:, :, 0], M,
                           f"Gland fraction: GT vs Pred ({base_method})")
            save_histogram(os.path.join(OUT_DIR, "hist_calc.png"),
                           pred_calc[:, :, 0], gt_calc[:, :, 0], M,
                           f"Calc fraction: GT vs Pred ({base_method})")

            gt_labels = np.zeros((Nx,Ny), dtype=np.uint8)
            gt_labels[gt_adip_bin] = 1
            gt_labels[gt_gland_bin] = 2
            gt_labels[gt_calc_bin] = 3

            save_scatter_mu(os.path.join(OUT_DIR, "scatter_mu40_mu60_gtlabels.png"),
                            mu40, mu60, gt_labels, M, nmax=60000)

    # spara alignment + mask-info
    np.savez(os.path.join(OUT_DIR, "eval_alignment_and_masks.npz"),
             best_A_name=best_name,
             best_A=best_A,
             best_shift=np.array([best_dy, best_dx], dtype=np.float32),
             inside_ref_mean=np.array([float(np.mean(inside_ref.astype(np.float32)))], dtype=np.float32),
             n_eval_pix=np.array([int(np.sum(M))], dtype=np.int32),
             )

    print("[INFO] Saved outputs to:", OUT_DIR)

    # Spara material-overlay för den aktiva slicen
    save_material_overlay(slice_idx=SLICE_IDX, out_dir=OUT_DIR, lpd_data_dir=LPD_DATA_DIR)

if __name__ == "__main__":
    main()

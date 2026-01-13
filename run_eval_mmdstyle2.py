# run_eval_mmdstyle.py
import os, json
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.ndimage import map_coordinates
from scipy.ndimage import shift as ndi_shift

# (valfritt) om du vill ha tabell som csv:
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
GT_TRANSPOSE = True
SAVE_DEBUG_IMAGES = True          # sparar overlay + dbg-mask (bara baseline)
SAVE_HISTS_SCATTER = False        # sätt True om du vill spara histogram/scatter för baseline

RECON_FOV_MM = 320.0
RECON_IMAGE_SIZE = 800

RECON_CENTER_OFFSET_MM   = (-15.0, 10.0)   # recon.centerOffset (x,y) mm
PHANTOM_CENTER_OFFSET_MM = (0.0, -70.0)    # phantom.centerOffset (x,y) mm

# ===== MANUELL FINJUSTERING (behåll din fungerande) =====
MANUAL_FINE_DY = -60.0   # upp
MANUAL_FINE_DX = +49.0   # vänster

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FDK_DIR = os.path.join(PROJECT_ROOT, "main", "EXE", "results", "xcist_fdk")
FDK40_NPY = os.path.join(FDK_DIR, "recon_fdk40.npy")
FDK60_NPY = os.path.join(FDK_DIR, "recon_fdk60.npy")

PHANTOM_DIR = os.path.join(PROJECT_ROOT, "xcist_voxelized_bundle_auto2d_calc", "UncompressedBreast1_z202")
GT_VOL1 = os.path.join(PHANTOM_DIR, "breast2d_vol_1.bin")  # adip
GT_VOL2 = os.path.join(PHANTOM_DIR, "breast2d_vol_2.bin")  # gland
GT_VOL3 = os.path.join(PHANTOM_DIR, "breast2d_vol_3.bin")  # skin
GT_VOL4 = os.path.join(PHANTOM_DIR, "breast2d_vol_4.bin")  # bone/calc
GT_JSON = os.path.join(PHANTOM_DIR, "breast_voxelized_2D.json")

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
    return float((np.mean(sig_vals) - np.mean(bg_vals)) / (np.sqrt(np.var(sig_vals) + np.var(bg_vals)) + eps))

# ============================================================
# Load GT once
# ============================================================
def load_gt_all():
    gt_shape = infer_square_2d_shape_from_bin(GT_VOL1, dtype=np.float32)
    gt_ad = read_bin(GT_VOL1, gt_shape, dtype=np.float32)
    gt_gl = read_bin(GT_VOL2, gt_shape, dtype=np.float32)
    gt_sk = read_bin(GT_VOL3, gt_shape, dtype=np.float32)
    gt_ca = read_bin(GT_VOL4, gt_shape, dtype=np.float32)

    gt_ad = maybe_transpose_gt(gt_ad, GT_TRANSPOSE)
    gt_gl = maybe_transpose_gt(gt_gl, GT_TRANSPOSE)
    gt_sk = maybe_transpose_gt(gt_sk, GT_TRANSPOSE)
    gt_ca = maybe_transpose_gt(gt_ca, GT_TRANSPOSE)

    gt_meta = load_gt_meta_from_json(GT_JSON)
    if GT_TRANSPOSE:
        gt_meta = gt_meta.copy()
        gt_meta["x_size"], gt_meta["y_size"] = gt_meta["y_size"], gt_meta["x_size"]
        gt_meta["x_offset"], gt_meta["y_offset"] = gt_meta["y_offset"], gt_meta["x_offset"]
        gt_meta["cols"], gt_meta["rows"] = gt_meta["rows"], gt_meta["cols"]

    return gt_ad, gt_gl, gt_sk, gt_ca, gt_meta

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
    gt_ad, gt_gl, gt_sk, gt_ca, gt_meta = load_gt_all()

    # ============================================================
    # INSIDE REF (gemensam mask för jämförelse)
    # ============================================================
    inside_ref = pick_inside_ref(recon_shape_xyz)
    pred_mask2d = inside_ref[:, :, 0].astype(bool)

    # ============================================================
    # ALIGNMENT (EN GÅNG) baserat på inside_ref
    # ============================================================
    best = None
    best_score = -1e9

    for name, A in candidate_A_matrices():
        tmp_ad = resample_gt_to_recon_affine(gt_ad[:, :, 0], gt_meta, (Nx, Ny), RECON_FOV_MM,
                                             RECON_CENTER_OFFSET_MM, PHANTOM_CENTER_OFFSET_MM, A, order=1)
        tmp_gl = resample_gt_to_recon_affine(gt_gl[:, :, 0], gt_meta, (Nx, Ny), RECON_FOV_MM,
                                             RECON_CENTER_OFFSET_MM, PHANTOM_CENTER_OFFSET_MM, A, order=1)
        tmp_ca = resample_gt_to_recon_affine(gt_ca[:, :, 0], gt_meta, (Nx, Ny), RECON_FOV_MM,
                                             RECON_CENTER_OFFSET_MM, PHANTOM_CENTER_OFFSET_MM, A, order=0)

        gt_mask_tmp = (tmp_ad + tmp_gl + tmp_ca) > 1e-6

        dy, dx, err = estimate_shift_com(pred_mask2d, gt_mask_tmp)
        gt_mask_aligned = shift2d_subpixel(gt_mask_tmp.astype(np.float32), dy=dy, dx=dx, order=0) > 0.5
        d = dice_mask(gt_mask_aligned, pred_mask2d)
        score = d - 0.01 * err

        if score > best_score:
            best_score = score
            best = (name, A, dy, dx, d, err)

    best_name, best_A, best_dy, best_dx, best_dice, best_err = best

    # manuell finjustering (din)
    best_dy = best_dy + MANUAL_FINE_DY
    best_dx = best_dx + MANUAL_FINE_DX

    run_tag = f"A{best_name}_dy{best_dy:.2f}_dx{best_dx:.2f}_man{MANUAL_FINE_DY}_{MANUAL_FINE_DX}"
    print("[INFO] Best A:", best_name, "dice=", best_dice, "shift(dy,dx)=", (best_dy, best_dx), "err=", best_err)

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
    gt_adip_bin  = gt_adip[:, :, 0]  > 0.5
    gt_gland_bin = gt_gland[:, :, 0] > 0.5
    gt_calc_bin  = gt_calc[:, :, 0]  > 0.1

    bg_for_adip  = gt_gland_bin
    bg_for_gland = gt_adip_bin
    bg_for_calc  = (gt_adip_bin | gt_gland_bin) & (~gt_calc_bin)

    sig_adip  = gt_adip_bin & M
    sig_gland = gt_gland_bin & M
    sig_calc  = gt_calc_bin & M

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

if __name__ == "__main__":
    main()

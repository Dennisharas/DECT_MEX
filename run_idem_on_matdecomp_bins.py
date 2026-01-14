import os
import argparse
import numpy as np 
import shutil


# -----------------------------
# I/O helpers for raw float32 .bin
# ----------------------------- 


def load_bin_float32(path: str, shape: tuple[int, int]) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = shape[0] * shape[1]
    if arr.size != expected:
        raise ValueError(
            f"{path}: expected {expected} float32 values for shape {shape}, got {arr.size}. "
            "Check --shape or whether the bin format includes a header."
        )
    return arr.reshape(shape)

def save_bin_float32(path: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.tofile(path)

def save_npy(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(arr, dtype=np.float32))

# -----------------------------
# IDEM core (your function, unchanged)
# -----------------------------
def _entropy_1d(vals: np.ndarray, n_bins: int = 256) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.inf
    h, _ = np.histogram(vals, bins=n_bins)
    p = h.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return np.inf
    p /= s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def _estimate_fwhm_major_peak(y: np.ndarray, n_bins: int = 256) -> float:
    y = y[np.isfinite(y)]
    if y.size < 100:
        return float(np.std(y) * 0.5 + 1e-6)

    hist, edges = np.histogram(y, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = int(np.argmax(hist))
    peak_val = hist[peak]
    if peak_val <= 0:
        return float(np.std(y) * 0.5 + 1e-6)

    half = peak_val / 2.0
    L = peak
    while L > 0 and hist[L] >= half:
        L -= 1
    R = peak
    while R < len(hist) - 1 and hist[R] >= half:
        R += 1

    fwhm = centers[R] - centers[L]
    if not np.isfinite(fwhm) or fwhm <= 0:
        fwhm = float(np.std(y) * 0.5 + 1e-6)
    return float(fwhm)

def _refine_axis_angle(points2xN: np.ndarray,
                       phi0: float,
                       search_deg: float = 5.0,
                       step_deg: float = 0.05,
                       n_bins: int = 256) -> float:
    best_H = np.inf
    best_psi = phi0
    for d in np.arange(-search_deg, search_deg + 1e-12, step_deg):
        psi = phi0 + np.deg2rad(float(d))
        axis = np.array([np.cos(psi), np.sin(psi)], dtype=np.float64)
        proj = axis @ points2xN
        H = _entropy_1d(proj, n_bins=n_bins)
        if H < best_H:
            best_H = H
            best_psi = psi
    return float(best_psi)

def idem_denoise_material_maps(
    x1: np.ndarray,
    x2: np.ndarray,
    img_low: np.ndarray | None = None,
    img_high: np.ndarray | None = None,
    beta: float = 1.0,
    delta: float = 15.0,
    sigma_hat: float | None = None,
    search_deg: float = 5.0,
    step_deg: float = 0.05,
    n_entropy_bins: int = 256,
    spatial_weighting: bool = False,
    eta: float = 1.0
):
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have same shape")

    H, W = x1.shape
    N = H * W

    if (img_low is None) != (img_high is None):
        raise ValueError("Provide both img_low and img_high, or neither.")
    if img_low is not None:
        img_low = np.asarray(img_low, dtype=np.float64)
        img_high = np.asarray(img_high, dtype=np.float64)
        if img_low.shape != x1.shape or img_high.shape != x1.shape:
            raise ValueError("img_low/img_high must match x1/x2 shape.")
        if sigma_hat is None:
            med = np.median(img_high)
            mad = np.median(np.abs(img_high - med)) + 1e-12
            sigma_hat = 1.4826 * mad

    X = np.stack([x1.ravel(), x2.ravel()], axis=0)

    # PCA-style initial direction (since we don't use A^{-1} here)
    C = np.cov(X)
    eigvals, eigvecs = np.linalg.eigh(C)
    v0 = eigvecs[:, np.argmax(eigvals)]
    phi0 = float(np.arctan2(v0[1], v0[0]))

    psi = _refine_axis_angle(X, phi0, search_deg=search_deg, step_deg=step_deg, n_bins=n_entropy_bins)

    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[ c,  s],
                  [-s,  c]], dtype=np.float64)
    Y = R @ X
    y1 = Y[0, :].copy()
    y2 = Y[1, :].copy()

    F = _estimate_fwhm_major_peak(y1, n_bins=256)
    width = float(beta * F + 1e-12)

    order = np.argsort(y1)
    y1s = y1[order]
    y2s = y2[order]

    if img_high is not None:
        low_s = img_low.ravel()[order]
        high_s = img_high.ravel()[order]

    if spatial_weighting:
        rr, cc = np.indices((H, W))
        rs = rr.ravel()[order].astype(np.float64)
        cs = cc.ravel()[order].astype(np.float64)

    y2s_new = y2s.copy()

    for i in range(N):
        center = y1s[i]
        L = np.searchsorted(y1s, center - width, side="left")
        Rr = np.searchsorted(y1s, center + width, side="right")
        if Rr - L <= 1:
            continue

        if img_high is not None:
            dh = high_s[L:Rr] - high_s[i]
            dl = low_s[L:Rr] - low_s[i]
            sig = float(delta * sigma_hat + 1e-12)
            w = np.exp(-(dh*dh + dl*dl) / (2.0 * sig*sig))
        else:
            w = np.ones((Rr - L,), dtype=np.float64)

        if spatial_weighting:
            dr = rs[L:Rr] - rs[i]
            dc = cs[L:Rr] - cs[i]
            D = np.sqrt(dr*dr + dc*dc)
            w *= (1.0 / (1.0 + (D / (eta + 1e-12))**2))

        sw = float(np.sum(w))
        if sw <= 0 or not np.isfinite(sw):
            continue

        y2s_new[i] = float(np.sum(w * y2s[L:Rr]) / sw)

    y2_new = np.empty_like(y2)
    y2_new[order] = y2s_new

    R_inv = np.array([[ c, -s],
                      [ s,  c]], dtype=np.float64)
    Y_new = np.stack([y1, y2_new], axis=0)
    X_new = R_inv @ Y_new

    x1_d = X_new[0, :].reshape(H, W).astype(np.float32)
    x2_d = X_new[1, :].reshape(H, W).astype(np.float32)

    info = {
        "psi_deg": float(np.rad2deg(psi)),
        "FWHM_y1": float(F),
        "width": float(width),
        "beta": float(beta),
        "delta": float(delta),
        "used_ct_weighting": img_high is not None,
    }
    return x1_d, x2_d, info

# -----------------------------
# Runner: read bins, run IDEM, write new bins
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        required=True,
        help="Folder containing frac_adip.bin, frac_gland.bin (and optionally frac_calc.bin)"
    )

    # out_dir tas bort som required – vi kan fortfarande tillåta override
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Optional override output folder. Default: <parent_of_in_dir>/image_matdecomp_DeNoise"
    )
    ap.add_argument(
        "--save_format",
        choices=["npy", "bin", "both"],
        default="npy",
        help="Output format: npy (default), bin, or both."
    )

    ap.add_argument(
        "--shape",
        required=True,
        nargs=2,
        type=int,
        metavar=("H", "W"),
        help="2D shape of each slice/bin (raw float32)."
    )

    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=15.0)
    ap.add_argument("--search_deg", type=float, default=5.0)
    ap.add_argument("--step_deg", type=float, default=0.05)
    ap.add_argument("--no_copy_calc", action="store_true", help="Do not copy frac_calc.bin if present.")

    # Om du vill att evaluation ska hitta samma filnamn (frac_adip.bin osv)
    ap.add_argument(
        "--overwrite_names",
        action="store_true",
        help="Save outputs as frac_adip.bin/frac_gland.bin (in the NEW out_dir). "
             "Default saves frac_adip_idem.bin/frac_gland_idem.bin."
    )

    args = ap.parse_args()

    H, W = args.shape
    in_dir = args.in_dir

    # --- AUTO out_dir: .../xcist_fdk/image_matdecomp_DeNoise
    if args.out_dir is None:
        parent = os.path.dirname(os.path.normpath(in_dir))
        out_dir = os.path.join(parent, "image_matdecomp_DeNoise")
    else:
        out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # ---- input files
    adip_path = os.path.join(in_dir, "frac_adip.bin")
    gland_path = os.path.join(in_dir, "frac_gland.bin")
    if not os.path.exists(adip_path) or not os.path.exists(gland_path):
        raise FileNotFoundError(f"Need frac_adip.bin and frac_gland.bin in: {in_dir}")

    frac_adip = load_bin_float32(adip_path, (H, W))
    frac_gland = load_bin_float32(gland_path, (H, W))

    # ---- IDEM
    adip_d, gland_d, info = idem_denoise_material_maps(
        frac_adip, frac_gland,
        img_low=None, img_high=None,
        beta=args.beta, delta=args.delta,
        search_deg=args.search_deg, step_deg=args.step_deg
    ) 

    # ---------------------------------------------------------
    # NEW: Force background to zero using inside_mask.bin
    # and copy inside_mask.bin to out_dir for evaluation
    # ---------------------------------------------------------
    inside_in = os.path.join(in_dir, "inside_mask.bin")
    inside_out = os.path.join(out_dir, "inside_mask.bin")

    if os.path.exists(inside_in):
        inside = load_bin_float32(inside_in, (H, W)) > 0.5
        

        # hard-zero outside breast
        adip_d[~inside]  = 0.0
        gland_d[~inside] = 0.0

        # optional but recommended: renormalize inside so adip+gland ~ 1
        s = adip_d + gland_d
        m = inside & (s > 1e-12)
        adip_d[m]  /= s[m]
        gland_d[m] /= s[m]

        # copy inside mask to output folder so evaluation uses it
        shutil.copyfile(inside_in, inside_out)

    else:
        print("[WARN] inside_mask.bin not found in in_dir, background may get small non-zero values after denoise.")


    # ---- output filenames
    # ---- output filenames (matcha eval-koden: f_*_raw.npy)
    adip_npy = os.path.join(out_dir, "f_adip_raw.npy")
    gland_npy = os.path.join(out_dir, "f_gland_raw.npy")

    # calc: vi har inte denoisat calc i den här 2-material IDEM-versionen,
    # så vi kopierar den (om den finns) och sparar som f_calc_raw.npy
    calc_in = os.path.join(in_dir, "frac_calc.bin")
    calc_arr = None
    if os.path.exists(calc_in) and not args.no_copy_calc:
        calc_arr = load_bin_float32(calc_in, (H, W))

    # If we have inside mask, also zero calc outside
    if calc_arr is not None and os.path.exists(inside_in):
        calc_arr[~inside] = 0.0

    # --- Save NPY
    if args.save_format in ["npy", "both"]:
        # Eval-koden förväntar sig ofta (Nx,Ny,1), så vi lägger till en z-dim
        save_npy(adip_npy, adip_d[:, :, None])
        save_npy(gland_npy, gland_d[:, :, None])
        if calc_arr is not None:
            save_npy(os.path.join(out_dir, "f_calc_raw.npy"), calc_arr[:, :, None])
        else:
            # om ingen calc finns, spara nollor (så eval inte crashar)
            save_npy(os.path.join(out_dir, "f_calc_raw.npy"), np.zeros((H, W, 1), dtype=np.float32))

    # --- Save BIN (om du vill behålla också)
    if args.save_format in ["bin", "both"]:
        if args.overwrite_names:
            adip_out_name = "frac_adip.bin"
            gland_out_name = "frac_gland.bin"
        else:
            adip_out_name = "frac_adip_idem.bin"
            gland_out_name = "frac_gland_idem.bin"

        save_bin_float32(os.path.join(out_dir, adip_out_name), adip_d)
        save_bin_float32(os.path.join(out_dir, gland_out_name), gland_d)

        if calc_arr is not None:
            save_bin_float32(os.path.join(out_dir, "frac_calc.bin"), calc_arr)

    # Save parameters
    with open(os.path.join(out_dir, "idem_info.txt"), "w") as f:
        f.write(f"in_dir: {in_dir}\n")
        f.write(f"out_dir: {out_dir}\n")
        f.write(f"save_format: {args.save_format}\n")
        for k, v in info.items():
            f.write(f"{k}: {v}\n")

        print("[IDEM] Read from:", in_dir)
        print("[IDEM] Saved to :", out_dir)
        print("[IDEM] info:", info)

if __name__ == "__main__":
    main()


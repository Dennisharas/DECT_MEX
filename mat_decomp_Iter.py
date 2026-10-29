import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import (
    gaussian_filter, median_filter,
    binary_erosion, binary_opening, binary_closing, binary_fill_holes,
    label
)

# ============================================================
# SETTINGS
# ============================================================
SIGMA_RECON = 3.5 
MEDIAN_SIZE = 3  


SKIN_THICKNESS = 6
CORNER_M = 60
BG_N_STD = 2.5

# Estimera calc-prototyp via residual/outliers:
CALC_RESID_PERCENTILE = 99.7   # ta topp outliers som calc-kandidater
CALC_PROTO_TOPK = 4000         # hur många punkter som används för calc-prototyp

# Kmeans för adip/gland centers
KMEANS_ITERS = 30
KMEANS_SUBSAMPLE = 120000

# Iterativ decomp
N_ITERS = 80
STEP = 0.35          # gradient step (kan behöva tweak: 0.1 - 0.6)
"""LAMBDA_A = 0.015     # TV strength adip
LAMBDA_G = 0.015     # TV strength gland
LAMBDA_C = 0.010     # TV strength calc (ofta lite lägre eller annorlunda)
TV_ITERS = 30        # Chambolle iterations (20-60 brukar vara ok)
""" 
LAMBDA_A = 0.006     # TV strength adip
LAMBDA_G = 0.006     # TV strength gland
LAMBDA_C = 0.003     # TV strength calc (ofta lite lägre eller annorlunda)
TV_ITERS = 10        # Chambolle iterations (20-60 brukar vara ok)
# NY 
DEBIAS_ITERS = 20
DO_DEBUG_IMAGES = True

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.join(PROJECT_ROOT, "DECT_Breast_Project", "UncompressedBreast1_CT", "z350", "xcist_fdk")

vol40_path = os.path.join(ROOT, "recon_fdk40.npy")
vol60_path = os.path.join(ROOT, "recon_fdk60.npy")

out_dir = os.path.join(ROOT, "image_matdecomp_iterative_tv")
os.makedirs(out_dir, exist_ok=True)

print("[DEBUG] PROJECT_ROOT =", PROJECT_ROOT)
print("[DEBUG] ROOT =", ROOT)
print("[DEBUG] vol40_path =", vol40_path)
print("[DEBUG] vol60_path =", vol60_path)

if not os.path.exists(vol40_path):
    raise FileNotFoundError(f"Hittar inte: {vol40_path}")
if not os.path.exists(vol60_path):
    raise FileNotFoundError(f"Hittar inte: {vol60_path}")

# ============================================================
# Helpers
# ============================================================
"""def save_img2d(img2d, fname, title, vmin=None, vmax=None, cmap="gray"):
    plt.figure(figsize=(5, 5))
    plt.imshow(img2d.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[INFO] Sparade", title, "->", out_path)
"""

def save_img2d(img2d, fname, title, vmin=None, vmax=None, cmap="gray"):
    img2d = np.asarray(img2d)

    # Om vmin/vmax inte ges: använd robust auto-scale (percentiler)
    if vmin is None or vmax is None:
        if np.any(np.isfinite(img2d)):
            # använd bara positiva pixlar om det finns (bra för fraktioner)
            if np.any(img2d > 0):
                vals = img2d[(img2d > 0) & np.isfinite(img2d)]
            else:
                vals = img2d[np.isfinite(img2d)]

            if vals.size > 0:
                vmin = float(np.percentile(vals, 1))
                vmax = float(np.percentile(vals, 99))
            else:
                vmin, vmax = None, None

    plt.figure(figsize=(5, 5))
    plt.imshow(img2d.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[INFO] Saved:", out_path, f"(vmin={vmin}, vmax={vmax})")

def largest_component(mask2d):
    lab, nlab = label(mask2d)
    if nlab <= 1:
        return mask2d
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = np.argmax(counts)
    return (lab == keep)

def kmeans2(X, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=2, replace=False)
    C = X[idx].copy()  # (2,2)
    for _ in range(iters):
        d0 = np.sum((X - C[0])**2, axis=1)
        d1 = np.sum((X - C[1])**2, axis=1)
        lab = (d1 < d0)
        if np.any(~lab):
            C[0] = X[~lab].mean(axis=0)
        if np.any(lab):
            C[1] = X[lab].mean(axis=0)
    return C

def project_fraction_to_line(P, A, G):
    v = (G - A).astype(np.float32)
    vv = float(v[0]*v[0] + v[1]*v[1]) + 1e-12
    t = ((P[:, 0] - A[0])*v[0] + (P[:, 1] - A[1])*v[1]) / vv
    return np.clip(t, 0.0, 1.0).astype(np.float32)

def residual_to_line(y40, y60, A, G, inside2):
    v = (G - A).astype(np.float32)
    vv = float(v[0]*v[0] + v[1]*v[1]) + 1e-12
    PA0 = (y40 - A[0]).astype(np.float32)
    PA1 = (y60 - A[1]).astype(np.float32)
    t = (PA0*v[0] + PA1*v[1]) / vv
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    y40_hat = A[0] + t*v[0]
    y60_hat = A[1] + t*v[1]
    res = np.sqrt((y40 - y40_hat)**2 + (y60 - y60_hat)**2).astype(np.float32)
    res = np.where(inside2, res, 0.0)
    return res

def project_to_simplex3(fa, fg, fc, inside2):
    """
    Projektion per pixel till simplex: f>=0, fa+fg+fc=1
    Snabb variant för 3 komponenter via sort/threshold.
    """
    # stack -> (H,W,3)
    F = np.stack([fa, fg, fc], axis=-1).astype(np.float32)
    # outside mask -> 0
    F[~inside2] = 0.0

    # reshape inside pixels
    idx = np.where(inside2.ravel())[0]
    if idx.size == 0:
        return fa*0, fg*0, fc*0

    Fin = F.reshape(-1, 3)[idx]  # (N,3)
    # standard simplex projection
    # sort desc
    U = np.sort(Fin, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1) - 1.0
    ind = np.array([1, 2, 3], dtype=np.float32)[None, :]
    cond = U - cssv / ind > 0
    rho = np.argmax(cond[:, ::-1], axis=1)
    rho = 2 - rho  # back to 0..2
    theta = cssv[np.arange(Fin.shape[0]), rho] / (rho + 1.0)
    Fin_proj = np.maximum(Fin - theta[:, None], 0.0)

    # write back
    Fout = F.reshape(-1, 3)
    Fout[idx] = Fin_proj
    F = Fout.reshape(F.shape)

    return F[..., 0], F[..., 1], F[..., 2]

def tv_denoise_chambolle(u, weight=0.02, n_iters=30, eps=1e-12):
    """
    Rudin-Osher-Fatemi TV denoise via Chambolle's projection algorithm (2D).
    weight ~ lambda. Större => mer smoothing.
    """
    u = u.astype(np.float32)
    px = np.zeros_like(u, dtype=np.float32)
    py = np.zeros_like(u, dtype=np.float32)

    tau = 0.25  # stable for 2D
    for _ in range(n_iters):
        # divergence
        div_p = (px - np.roll(px, 1, axis=0)) + (py - np.roll(py, 1, axis=1))
        # update
        v = div_p - (u / weight)
        # gradients of v
        vx = np.roll(v, -1, axis=0) - v
        vy = np.roll(v, -1, axis=1) - v
        # update p
        px_new = px + tau * vx
        py_new = py + tau * vy
        norm = np.maximum(1.0, np.sqrt(px_new**2 + py_new**2) + eps)
        px = px_new / norm
        py = py_new / norm

    div_p = (px - np.roll(px, 1, axis=0)) + (py - np.roll(py, 1, axis=1))
    out = u - weight * div_p
    return out.astype(np.float32)

# ============================================================
# Load recon
# ============================================================
vol40 = np.load(vol40_path).astype(np.float32)
vol60 = np.load(vol60_path).astype(np.float32)

if vol40.shape != vol60.shape:
    raise RuntimeError(f"Volymerna har olika shape: {vol40.shape} vs {vol60.shape}")

nx, ny, nz = vol40.shape
zmid = nz // 2
print(f"[INFO] Recon shape: {vol40.shape}")

diff_norm = float(np.linalg.norm((vol40 - vol60).ravel()) / (np.linalg.norm(vol60.ravel()) + 1e-12))
print(f"[DEBUG] rel-norm(vol40-vol60)/norm(vol60) = {diff_norm:.3e}")
if diff_norm < 1e-6:
    print("[WARN] vol40 och vol60 ser nästan identiska ut. Kontrollera att du inte råkat använda samma energi!")

# ============================================================
# UNIT sanity (1/mm vs 1/cm)
# ============================================================
im40 = vol40[:, :, zmid]
thr_inside = np.percentile(im40, 20)
tmp_mask2d = im40 > thr_inside
med40 = float(np.median(im40[tmp_mask2d]))
print("[DEBUG] median(vol40 inside-ish) =", med40)

UNIT_SCALE = 1.0
if med40 < 0.08:
    UNIT_SCALE = 10.0
    print("[WARN] Recon verkar vara i 1/mm. Skalar *10 -> 1/cm.")
else:
    print("[INFO] Recon verkar vara i 1/cm. Ingen skalning.")
vol40 *= UNIT_SCALE
vol60 *= UNIT_SCALE

# ============================================================
# Smooth recon (robusthet)
# ============================================================
vol40_s = gaussian_filter(vol40, sigma=SIGMA_RECON)
vol60_s = gaussian_filter(vol60, sigma=SIGMA_RECON)

vol40_s = median_filter(vol40_s, size=MEDIAN_SIZE)
vol60_s = median_filter(vol60_s, size=MEDIAN_SIZE)

y40 = vol40_s[:, :, zmid]
y60 = vol60_s[:, :, zmid]

# ============================================================
# Inside mask (mid-slice) - samma idé som du använder
# ============================================================
m = CORNER_M
corners = np.concatenate([
    y40[:m, :m].ravel(),
    y40[:m, -m:].ravel(),
    y40[-m:, :m].ravel(),
    y40[-m:, -m:].ravel(),
])
bg_med = np.median(corners)
bg_mad = np.median(np.abs(corners - bg_med)) + 1e-12
bg_std = 1.4826 * bg_mad
thr = bg_med + BG_N_STD * bg_std

body2 = (y40 > thr)
body2 = binary_opening(body2, iterations=1)
body2 = binary_closing(body2, iterations=2)
body2 = binary_fill_holes(body2)
body2 = largest_component(body2)

inside2 = binary_erosion(body2, iterations=SKIN_THICKNESS)
if inside2.sum() == 0:
    print("[WARN] Erosion gav tom mask. Sätter SKIN_THICKNESS=2.")
    inside2 = binary_erosion(body2, iterations=2)

print("[DEBUG] inside2 sum:", int(inside2.sum()))
if DO_DEBUG_IMAGES:
    save_img2d(body2.astype(np.float32), "debug_body2.png", "debug: body2", vmin=0, vmax=1)
    save_img2d(inside2.astype(np.float32), "debug_inside2.png", "debug: inside2", vmin=0, vmax=1)

# ============================================================
# 1) Skatta adip/gland centers via kmeans i (y40,y60)-rymden
# ============================================================
P = np.stack([y40[inside2], y60[inside2]], axis=1)

# ta bort extrema toppar innan kmeans
hi40 = np.percentile(P[:, 0], 99.7)
hi60 = np.percentile(P[:, 1], 99.7)
keep = (P[:, 0] < hi40) & (P[:, 1] < hi60)
P_fit = P[keep]

if P_fit.shape[0] < 2000:
    raise RuntimeError("För få punkter för kmeans. Kolla masken.")

if P_fit.shape[0] > KMEANS_SUBSAMPLE:
    rng = np.random.default_rng(0)
    idx = rng.choice(P_fit.shape[0], size=KMEANS_SUBSAMPLE, replace=False)
    P_train = P_fit[idx]
else:
    P_train = P_fit

C = kmeans2(P_train, iters=KMEANS_ITERS, seed=0)

# label adip vs gland: adip lägre attenuation (y40+y60)
s0 = float(C[0, 0] + C[0, 1])
s1 = float(C[1, 0] + C[1, 1])
if s0 <= s1:
    A = C[0].astype(np.float32)
    G = C[1].astype(np.float32)
else:
    A = C[1].astype(np.float32)
    G = C[0].astype(np.float32)

print("[INFO] centers:")
print("  A(adip) =", A)
print("  G(gland)=", G)

# ============================================================
# 2) Skatta calc-prototyp (Ccalc) från residual outliers
# ============================================================
res = residual_to_line(y40, y60, A, G, inside2)
res_vals = res[inside2]
thr_res = float(np.percentile(res_vals, CALC_RESID_PERCENTILE))
cand = inside2 & (res >= thr_res)

cand_pts = np.stack([y40[cand], y60[cand]], axis=1)
if cand_pts.shape[0] < 50:
    print("[WARN] Få calc-kandidater. Använder max-intensitet som fallback.")
    # fallback: topp-punkter i (y40+y60)
    S = (y40 + y60)
    flat = S[inside2].ravel()
    tS = np.percentile(flat, 99.8)
    cand = inside2 & (S >= tS)
    cand_pts = np.stack([y40[cand], y60[cand]], axis=1)

# ta topp K och robust median
if cand_pts.shape[0] > CALC_PROTO_TOPK:
    rng = np.random.default_rng(1)
    idx = rng.choice(cand_pts.shape[0], size=CALC_PROTO_TOPK, replace=False)
    cand_pts = cand_pts[idx]

Ccalc = np.median(cand_pts, axis=0).astype(np.float32)
print("[INFO] calc prototype (data-driven) =", Ccalc)

if DO_DEBUG_IMAGES:
    save_img2d(res, "debug_residual.png", "debug: residual", vmin=0)

# ============================================================
# Bygg mixing matrix M (2x3): [A, G, Ccalc]
# ============================================================
M = np.stack([A, G, Ccalc], axis=1).astype(np.float32)  # shape (2,3)
print("[INFO] M =\n", M)

# ============================================================
# Init f med din "line projection" (adip/gland) + lite calc-seed
# ============================================================
P_all = np.stack([y40.ravel(), y60.ravel()], axis=1)
t_all = project_fraction_to_line(P_all, A, G).reshape(nx, ny)

fa = np.zeros((nx, ny), dtype=np.float32)
fg = np.zeros((nx, ny), dtype=np.float32)
fc = np.zeros((nx, ny), dtype=np.float32)

fg[inside2] = t_all[inside2]
fa[inside2] = 1.0 - fg[inside2]

# seed calc lite där residual är stor (soft)
# normalisera residual till [0,1] inom mask och skala ner
rv = res_vals
rmin, rmax = float(np.min(rv)), float(np.max(rv) + 1e-12)
r01 = (res - rmin) / (rmax - rmin)
seed = np.clip((r01 - 0.7) / 0.3, 0.0, 1.0)  # bara hög residual ger seed
fc[inside2] = 0.10 * seed[inside2]            # max 10% calc-seed

# renormalisera adip/gland när calc seed läggs
fa = fa * (1.0 - fc)
fg = fg * (1.0 - fc)

# project to simplex
fa, fg, fc = project_to_simplex3(fa, fg, fc, inside2)

# ============================================================
# Iterativ optimering
# ============================================================
# Precompute for gradient: MtM, Mty
Mt = M.T  # (3,2)
MtM = (Mt @ M).astype(np.float32)  # (3,3)

# y per pixel: shape (2, nx, ny)
Y0 = y40.astype(np.float32)
Y1 = y60.astype(np.float32)

print("[INFO] Startar iterationer:", N_ITERS)
for it in range(N_ITERS):
    # ---- gradient of data term: grad = 2 * M^T (M f - y)
    # compute Mf per pixel:
    # yhat0 = M00*fa + M01*fg + M02*fc
    # yhat1 = M10*fa + M11*fg + M12*fc
    yhat0 = M[0, 0]*fa + M[0, 1]*fg + M[0, 2]*fc
    yhat1 = M[1, 0]*fa + M[1, 1]*fg + M[1, 2]*fc

    e0 = (yhat0 - Y0)
    e1 = (yhat1 - Y1)

    # grad components
    gfa = 2.0*(M[0, 0]*e0 + M[1, 0]*e1)
    gfg = 2.0*(M[0, 1]*e0 + M[1, 1]*e1)
    gfc = 2.0*(M[0, 2]*e0 + M[1, 2]*e1)

    # only inside updates
    gfa = np.where(inside2, gfa, 0.0).astype(np.float32)
    gfg = np.where(inside2, gfg, 0.0).astype(np.float32)
    gfc = np.where(inside2, gfc, 0.0).astype(np.float32)

    # gradient step
    fa = fa - STEP * gfa
    fg = fg - STEP * gfg
    fc = fc - STEP * gfc

    # proximal TV per channel (denoise) ---> Gammal kod
    """if LAMBDA_A > 0:
        fa = tv_denoise_chambolle(fa, weight=LAMBDA_A, n_iters=TV_ITERS)
    if LAMBDA_G > 0:
        fg = tv_denoise_chambolle(fg, weight=LAMBDA_G, n_iters=TV_ITERS)
    if LAMBDA_C > 0:
        fc = tv_denoise_chambolle(fc, weight=LAMBDA_C, n_iters=TV_ITERS)
    """ 
    if it < (N_ITERS - DEBIAS_ITERS):
        if LAMBDA_A > 0:
            fa = tv_denoise_chambolle(fa, weight=LAMBDA_A, n_iters=TV_ITERS)
        if LAMBDA_G > 0:
            fg = tv_denoise_chambolle(fg, weight=LAMBDA_G, n_iters=TV_ITERS)
        if LAMBDA_C > 0:
            fc = tv_denoise_chambolle(fc, weight=LAMBDA_C, n_iters=TV_ITERS)

    # project to simplex (nonneg + sum=1)
    fa, fg, fc = project_to_simplex3(fa, fg, fc, inside2)

    # debug / logging
    if (it % 10) == 0 or it == (N_ITERS - 1):
        # data fidelity measure inside
        err = np.sqrt((e0[inside2]**2 + e1[inside2]**2).mean())
        print(f"[ITER {it:03d}] rmse(data)={float(err):.4e}   mean(fc)={float(fc[inside2].mean()):.4f}")

# ============================================================
# Outputs: raw basis maps (2D) + pack into 3D with only mid-slice
# ============================================================
f_adip_raw  = fa.astype(np.float32)
f_gland_raw = fg.astype(np.float32)
f_calc_raw  = fc.astype(np.float32)

# save images
save_img2d(f_adip_raw,  "f_adip_raw.png",  "f_adip_raw (iter+TV)",  vmin=0, vmax=1)
save_img2d(f_gland_raw, "f_gland_raw.png", "f_gland_raw (iter+TV)", vmin=0, vmax=1)
save_img2d(f_calc_raw,  "f_calc_raw.png",  "f_calc_raw (iter+TV)",  vmin=0, vmax=1)

# Make 3D volumes (only zmid filled) for compatibility
frac_adip  = np.zeros((nx, ny, nz), dtype=np.float32)
frac_gland = np.zeros((nx, ny, nz), dtype=np.float32)
frac_calc  = np.zeros((nx, ny, nz), dtype=np.float32)

frac_adip[:, :, zmid]  = f_adip_raw
frac_gland[:, :, zmid] = f_gland_raw
frac_calc[:, :, zmid]  = f_calc_raw

np.save(os.path.join(out_dir, "f_adip_raw.npy"),  f_adip_raw)
np.save(os.path.join(out_dir, "f_gland_raw.npy"), f_gland_raw)
np.save(os.path.join(out_dir, "f_calc_raw.npy"),  f_calc_raw)

np.save(os.path.join(out_dir, "frac_adip.npy"),  frac_adip)
np.save(os.path.join(out_dir, "frac_gland.npy"), frac_gland)
np.save(os.path.join(out_dir, "frac_calc.npy"),  frac_calc)

np.save(os.path.join(out_dir, "inside_mask.npy"), inside2.astype(np.uint8))

# Save bin float32
def save_bin(path, arr):
    np.asarray(arr, dtype=np.float32).tofile(path)
    print("[INFO] Sparade BIN ->", path)

save_bin(os.path.join(out_dir, "frac_adip.bin"),  frac_adip)
save_bin(os.path.join(out_dir, "frac_gland.bin"), frac_gland)
save_bin(os.path.join(out_dir, "frac_calc.bin"),  frac_calc)

print("[INFO] Klart! Output:", out_dir)

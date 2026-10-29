import os
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import median_filter 
from scipy.ndimage import binary_dilation



from scipy.ndimage import (
    gaussian_filter,
    binary_erosion, binary_opening, binary_closing, binary_fill_holes,
    binary_dilation,
    label
)

# ============================================================
# SETTINGS
# ============================================================
SIGMA_RECON = 3.5          # smoothing på recon (INNAN analys) 0.8 innan 
SKIN_THICKNESS = 6
CORNER_M = 60
BG_N_STD = 2.5

# calc-detektion via residual (procentil av residual inne i bröstet)
T_CALC_PERCENTILE = 99.9

# kmeans (2 kluster) för adip/gland
KMEANS_ITERS = 30
KMEANS_SUBSAMPLE = 120000   # antal punkter att träna på (för speed)

# optional: gör calc-mask lite tjockare (för att fånga hela kalken)
CALC_DILATE_ITERS = 1

# Output
DO_DEBUG_IMAGES = True

# ============================================================
# PATHS (FDK) - EXAKT platsen du vill använda
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.join(PROJECT_ROOT, "DECT_Breast_Project" ,"UncompressedBreast1_CT" ,"z350" , "xcist_fdk")

vol40_path = os.path.join(ROOT, "recon_fdk40.npy")
vol60_path = os.path.join(ROOT, "recon_fdk60.npy")

out_dir = os.path.join(ROOT, "image_matdecomp_datadriven")
os.makedirs(out_dir, exist_ok=True)

print("[DEBUG] PROJECT_ROOT =", PROJECT_ROOT)
print("[DEBUG] ROOT (xcist_fdk) =", ROOT)
print("[DEBUG] vol40_path =", vol40_path)
print("[DEBUG] vol60_path =", vol60_path)

if not os.path.exists(vol40_path):
    raise FileNotFoundError(f"Hittar inte: {vol40_path}")
if not os.path.exists(vol60_path):
    raise FileNotFoundError(f"Hittar inte: {vol60_path}")

# ============================================================
# Helpers
# ============================================================
def save_img2d(img2d, fname, title, vmin=None, vmax=None, cmap="gray"):
    plt.figure(figsize=(5, 5))
    plt.imshow(img2d.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[INFO] Sparade", title, "->", out_path)

def save_slice3d(vol3d, fname, title, vmin=None, vmax=None, cmap="gray"):
    zmid = vol3d.shape[2] // 2
    save_img2d(vol3d[:, :, zmid], fname, title, vmin=vmin, vmax=vmax, cmap=cmap)

def largest_component(mask2d):
    lab, nlab = label(mask2d)
    if nlab <= 1:
        return mask2d
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = np.argmax(counts)
    return (lab == keep)

def kmeans2(X, iters=20, seed=0):
    """
    X: (N,2)
    return centers (2,2)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=2, replace=False)
    C = X[idx].copy()  # (2,2)

    for _ in range(iters):
        # assign
        d0 = np.sum((X - C[0])**2, axis=1)
        d1 = np.sum((X - C[1])**2, axis=1)
        lab = (d1 < d0)

        # update
        if np.any(~lab):
            C[0] = X[~lab].mean(axis=0)
        if np.any(lab):
            C[1] = X[lab].mean(axis=0)

    return C

def project_fraction_to_line(P, A, G):
    """
    P: (N,2) points
    A,G: (2,) endpoints
    returns t in [0,1] where t=0 => A, t=1 => G
    """
    v = (G - A).astype(np.float32)
    vv = float(v[0]*v[0] + v[1]*v[1]) + 1e-12
    t = ((P[:, 0] - A[0])*v[0] + (P[:, 1] - A[1])*v[1]) / vv
    return np.clip(t, 0.0, 1.0).astype(np.float32)

# ============================================================
# Load recon
# ============================================================
print("[INFO] Läser FDK-volymer:")
print("  40 kVp:", vol40_path)
print("  60 kVp:", vol60_path)

vol40 = np.load(vol40_path).astype(np.float32)
vol60 = np.load(vol60_path).astype(np.float32)

if vol40.shape != vol60.shape:
    raise RuntimeError(f"Volymerna har olika shape: {vol40.shape} vs {vol60.shape}")

nx, ny, nz = vol40.shape
zmid = nz // 2
print(f"[INFO] Recon shape: {vol40.shape}, Nvox={nx*ny*nz}")

diff_norm = float(np.linalg.norm((vol40 - vol60).ravel()) / (np.linalg.norm(vol60.ravel()) + 1e-12))
print(f"[DEBUG] rel-norm(vol40-vol60)/norm(vol60) = {diff_norm:.3e}")

# ============================================================
# UNIT sanity (mm^-1 vs cm^-1)  (din gamla check är okej)
# ============================================================
im40 = vol40[:, :, zmid]
thr_inside = np.percentile(im40, 20)
tmp_mask2d = im40 > thr_inside

med40 = float(np.median(im40[tmp_mask2d]))
med60 = float(np.median(vol60[:, :, zmid][tmp_mask2d]))
print("[DEBUG] median(vol40 inside-ish) =", med40)
print("[DEBUG] median(vol60 inside-ish) =", med60)

UNIT_SCALE = 1.0
if med40 < 0.08:
    UNIT_SCALE = 10.0
    print("[WARN] Recon verkar vara i 1/mm. Skalar volymer *10 för att få 1/cm.")
else:
    print("[INFO] Recon verkar redan vara i 1/cm. Ingen skalning.")

vol40 *= UNIT_SCALE
vol60 *= UNIT_SCALE

# Smooth recon (endast för robusthet)
vol40_s = gaussian_filter(vol40, sigma=SIGMA_RECON)
vol60_s = gaussian_filter(vol60, sigma=SIGMA_RECON)

# NEW: median filter (try size=3 first)
vol40_s = median_filter(vol40_s, size=3)
vol60_s = median_filter(vol60_s, size=3) 

# ============================================================
# Inside mask (air/skin bort) på 2D-slice
# ============================================================
im40s = vol40_s[:, :, zmid]
m = CORNER_M
corners = np.concatenate([
    im40s[:m, :m].ravel(),
    im40s[:m, -m:].ravel(),
    im40s[-m:, :m].ravel(),
    im40s[-m:, -m:].ravel(),
])
bg_med = np.median(corners)
bg_mad = np.median(np.abs(corners - bg_med)) + 1e-12
bg_std = 1.4826 * bg_mad
thr = bg_med + BG_N_STD * bg_std

body2 = (im40s > thr)
body2 = binary_opening(body2, iterations=1)
body2 = binary_closing(body2, iterations=2)
body2 = binary_fill_holes(body2)
body2 = largest_component(body2)

eroded = binary_erosion(body2, iterations=SKIN_THICKNESS)
if eroded.sum() == 0:
    print("[WARN] Erosion gav tom mask. Sätter SKIN_THICKNESS=2.")
    eroded = binary_erosion(body2, iterations=2)

inside2 = eroded  # robust: "inside" = eroderad body

inside_mask = np.zeros_like(vol40_s, dtype=bool)
inside_mask[:, :, zmid] = inside2

print("[DEBUG] bg_med, bg_std, thr:", float(bg_med), float(bg_std), float(thr))
print("[DEBUG] inside2 sum:", int(inside2.sum()))

if DO_DEBUG_IMAGES:
    save_img2d(body2.astype(np.float32), "debug_body2.png", "debug: body2", vmin=0, vmax=1)
    save_img2d(inside2.astype(np.float32), "debug_inside2.png", "debug: inside2", vmin=0, vmax=1)

# ============================================================
# DATA-DRIVEN adip/gland: kmeans i (y40,y60) rymd
#   1) skapa prelim calc/outlier mask via residual till PCA-linje-ish
#      (enklare: använd högintensiva outliers i vol40+vol60)
# ============================================================
y40_2d = vol40_s[:, :, zmid]
y60_2d = vol60_s[:, :, zmid]

inside_pts = inside2
P = np.stack([y40_2d[inside_pts], y60_2d[inside_pts]], axis=1)  # (N,2)

# ta bort extrema toppar (ofta calc/metall) innan vi hittar adip/gland-centers
hi40 = np.percentile(P[:, 0], 99.7)
hi60 = np.percentile(P[:, 1], 99.7)
keep = (P[:, 0] < hi40) & (P[:, 1] < hi60)
P_fit = P[keep]

if P_fit.shape[0] < 2000:
    raise RuntimeError("För få punkter kvar för att skatta adip/gland. Kolla inside-masken.")

# subsample för speed
if P_fit.shape[0] > KMEANS_SUBSAMPLE:
    rng = np.random.default_rng(0)
    idx = rng.choice(P_fit.shape[0], size=KMEANS_SUBSAMPLE, replace=False)
    P_train = P_fit[idx]
else:
    P_train = P_fit

C = kmeans2(P_train, iters=KMEANS_ITERS, seed=0)  # (2,2)

# bestäm vilken center som är adip vs gland:
# adip har typiskt lägre attenuation => lägre (y40+y60)
s0 = float(C[0, 0] + C[0, 1])
s1 = float(C[1, 0] + C[1, 1])
if s0 <= s1:
    A = C[0]
    G = C[1]
else:
    A = C[1]
    G = C[0]

print("[INFO] kmeans centers:")
print("       A(adip)  =", A)
print("       G(gland) =", G)

# ============================================================
# Fractioner från projektion på linjen A->G
# ============================================================
# för hela 2D-slice:
P_all = np.stack([y40_2d.ravel(), y60_2d.ravel()], axis=1)
t_all = project_fraction_to_line(P_all, A, G).reshape(nx, ny)  # gland-fraction i 2D
f_gland_2d = np.zeros((nx, ny), dtype=np.float32)
f_adip_2d  = np.zeros((nx, ny), dtype=np.float32)

f_gland_2d[inside2] = t_all[inside2]
f_adip_2d[inside2]  = 1.0 - f_gland_2d[inside2]

# ============================================================
# Calc via residual till A->G-linjen i 2D
# ============================================================
v = (G - A).astype(np.float32)
vv = float(v[0]*v[0] + v[1]*v[1]) + 1e-12

# projektion -> närmaste punkt på linjen
PA0 = (y40_2d - A[0]).astype(np.float32)
PA1 = (y60_2d - A[1]).astype(np.float32)
t = (PA0*v[0] + PA1*v[1]) / vv
t = np.clip(t, 0.0, 1.0).astype(np.float32)

y40_hat = A[0] + t*v[0]
y60_hat = A[1] + t*v[1]
res2d = np.sqrt((y40_2d - y40_hat)**2 + (y60_2d - y60_hat)**2).astype(np.float32)
res2d = np.where(inside2, res2d, 0.0)

res_vals = res2d[inside2]
t_calc = float(np.percentile(res_vals, T_CALC_PERCENTILE))
calc2d = (res2d > t_calc) & inside2

# gör calc lite tjockare om du vill (ofta bättre visuellt)
if CALC_DILATE_ITERS > 0:
    calc2d = binary_dilation(calc2d, iterations=CALC_DILATE_ITERS)

noncalc2d = inside2 & (~calc2d)

print("[DEBUG] calc threshold(res) =", t_calc, " calc pixels =", int(calc2d.sum()))

# ============================================================
# Slutliga fraction maps (3D volymer med nz=1)
#   Viktigt: INGEN gaussian_filter här => inga halos.
# ============================================================
frac_adip  = np.zeros((nx, ny, nz), dtype=np.float32)
frac_gland = np.zeros((nx, ny, nz), dtype=np.float32)
frac_calc  = np.zeros((nx, ny, nz), dtype=np.float32)

frac_adip[:, :, zmid][noncalc2d]  = f_adip_2d[noncalc2d]
frac_gland[:, :, zmid][noncalc2d] = f_gland_2d[noncalc2d]
frac_calc[:, :, zmid][calc2d]     = 1.0

# (valfritt) säkerställ adip+gland=1 i noncalc
s = frac_adip[:, :, zmid] + frac_gland[:, :, zmid] + 1e-12
frac_adip[:, :, zmid][noncalc2d]  = (frac_adip[:, :, zmid][noncalc2d]  / s[noncalc2d]).astype(np.float32)
frac_gland[:, :, zmid][noncalc2d] = (frac_gland[:, :, zmid][noncalc2d] / s[noncalc2d]).astype(np.float32)

# ============================================================
# Save outputs
# ============================================================
save_slice3d(frac_adip,  "frac_adip_mid.png",  "Frac adip (data-driven, mid)",  vmin=0.0, vmax=1.0)
save_slice3d(frac_gland, "frac_gland_mid.png", "Frac gland (data-driven, mid)", vmin=0.0, vmax=1.0)
save_slice3d(frac_calc,  "frac_calc_mid.png",  "Frac calc (data-driven, mid)",  vmin=0.0, vmax=1.0)

np.save(os.path.join(out_dir, "frac_adip.npy"),  frac_adip)
np.save(os.path.join(out_dir, "frac_gland.npy"), frac_gland)
np.save(os.path.join(out_dir, "frac_calc.npy"),  frac_calc)
np.save(os.path.join(out_dir, "calc_mask.npy"),  calc2d.astype(np.uint8))
np.save(os.path.join(out_dir, "inside_mask.npy"), inside2.astype(np.uint8))


# ============================================================
# (OPTIONAL) Save as .bin (float32) for compatibility
# ============================================================
def save_bin(path, arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr.tofile(path)
    print("[INFO] Sparade BIN ->", path)

save_bin(os.path.join(out_dir, "frac_adip.bin"),  frac_adip)
save_bin(os.path.join(out_dir, "frac_gland.bin"), frac_gland)
save_bin(os.path.join(out_dir, "frac_calc.bin"),  frac_calc)

# Om du vill spara masks som uint8:
calc_mask_3d = np.zeros((nx, ny, nz), dtype=np.uint8)
calc_mask_3d[:, :, zmid] = calc2d.astype(np.uint8)
save_bin(os.path.join(out_dir, "calc_mask.bin"), calc_mask_3d.astype(np.float32))  # eller uint8 om du vill

inside_mask_3d = np.zeros((nx, ny, nz), dtype=np.uint8)
inside_mask_3d[:, :, zmid] = inside2.astype(np.uint8)
save_bin(os.path.join(out_dir, "inside_mask.bin"), inside_mask_3d.astype(np.float32))

if DO_DEBUG_IMAGES:
    save_img2d(res2d, "debug_residual.png", "debug: residual to A-G line")
    save_img2d(calc2d.astype(np.float32), "debug_calc_mask.png", "debug: calc_mask", vmin=0, vmax=1)
    save_img2d(f_gland_2d, "debug_fgland_raw.png", "debug: f_gland_raw", vmin=0, vmax=1)
    save_img2d(f_adip_2d, "debug_fadip_raw.png", "debug: f_adip_raw", vmin=0, vmax=1)

print("[INFO] Klart! Output:", out_dir)



























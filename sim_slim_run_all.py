import os, re
import numpy as np

import gecatsim as xc
from gecatsim.pyfiles import CommonTools as CT

# ----------------- inställningar -----------------
ENERGY_TAG = "60kVp_Cu0.2"

ALLZ_DIR = os.path.expanduser(
    "~/DECT_Breast_Project/xcist_voxelized_bundle_auto2d_calc/UncompressedBreast1_allz2"
)

OUTROOT = os.path.expanduser(
    "~/DECT_Breast_Project/UncompressedBreast1_CT"
)

THRESH = 0.2                # tröskel i center-of-mass på sino-raden
OFFSET_INDEX = 0            # 0=x, 1=y, 2=z
# -------------------------------------------------


# --- robust hitta projektroot och main/gecatsim ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))               # .../main/EXE
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # .../DECT_Breast_Project
PKG = os.path.join(PROJECT_ROOT, "main", "gecatsim")
CFG_DIR = os.path.join(PKG, "cfg")
PHANTOM_CFG = os.path.join(CFG_DIR, "Phantom_Default.cfg")

if not os.path.exists(PHANTOM_CFG):
    raise FileNotFoundError(
        f"Hittar inte Phantom_Default.cfg här:\n  {PHANTOM_CFG}\n"
        f"Kollade PROJECT_ROOT={PROJECT_ROOT}"
    )

CT.my_path.paths = {
    "cfg"     : CFG_DIR,
    "spectrum": os.path.join(PKG, "spectrum"),
    "bowtie"  : os.path.join(PKG, "bowtie"),
    "material": os.path.join(PKG, "material"),
    "phantom" : os.path.join(PKG, "phantom"),
    "lib"     : os.path.join(PKG, "lib"),
}

def list_z_dirs(root):
    out = []
    for name in os.listdir(root):
        m = re.match(r"z(\d+)$", name)
        if m:
            out.append((int(m.group(1)), os.path.join(root, name)))
    return [(zi, p) for zi, p in sorted(out)]

def read_centeroffset_from_cfg(text):
    m = re.search(r"^\s*phantom\.centerOffset\s*=\s*\[([^\]]+)\]", text, flags=re.M)
    if not m:
        return None
    nums = [float(x.strip()) for x in m.group(1).split(",")]
    return nums if len(nums) == 3 else None

def write_centeroffset_to_cfg(path, new_offset):
    txt = open(path, "r").read()
    if read_centeroffset_from_cfg(txt) is None:
        raise RuntimeError("Hittar ingen rad 'phantom.centerOffset = [...]' i Phantom_Default.cfg")

    repl = f"phantom.centerOffset = [{new_offset[0]:.6f}, {new_offset[1]:.6f}, {new_offset[2]:.6f}]"
    txt2 = re.sub(
        r"^\s*phantom\.centerOffset\s*=\s*\[[^\]]+\]",
        repl,
        txt,
        flags=re.M
    )
    open(path, "w").write(txt2)

def ensure_prep(prefix, ct_obj):
    """Om .prep inte skapas av din build: bygg den från .scan/.air."""
    prep_path = prefix + ".prep"
    if os.path.exists(prep_path):
        return prep_path

    scan_path = prefix + ".scan"
    air_path  = prefix + ".air"
    if not (os.path.exists(scan_path) and os.path.exists(air_path)):
        return None

    cfg = ct_obj.get_current_cfg()
    v = int(cfg.protocol.viewCount)
    r = int(cfg.scanner.detectorRowCount)
    c = int(cfg.scanner.detectorColCount)

    air_cr   = CT.rawread(air_path,  [c, r],    "float")         # (c,r)
    scan_crv = CT.rawread(scan_path, [c, r, v], "float")         # (c,r,v)

    air_rc   = air_cr.T                                          # (r,c)
    scan_vrc = np.transpose(scan_crv, (2, 1, 0))                 # (v,r,c)

    eps = 1e-6
    p_vrc = -np.log(np.clip(scan_vrc, eps, None) / np.clip(air_rc, eps, None))
    p_vrc.astype(np.float32, copy=False).tofile(prep_path)
    return prep_path

def compute_dmm_from_prep(prep_path, cfg, threshold=0.2):
    V = int(cfg.protocol.viewCount)
    R = int(cfg.scanner.detectorRowCount)
    C = int(cfg.scanner.detectorColCount)

    SID = float(cfg.scanner.sid)
    SDD = float(cfg.scanner.sdd)
    pix_mm = float(cfg.scanner.detectorColSize)

    prep = CT.rawread(prep_path, [V, R, C], "float")
    sino = prep[:, R//2, :]  # (V,C)

    det_center = (C - 1) / 2.0
    cols = np.arange(C, dtype=float)
    centers = np.full(V, np.nan, dtype=float)

    for v in range(V):
        row = sino[v, :]
        mask = row > threshold
        if not np.any(mask):
            continue
        w = row[mask]
        c = cols[mask]
        centers[v] = (c*w).sum() / (w.sum() + 1e-12)

    mean_center = np.nanmean(centers)
    delta_u_pix = mean_center - det_center
    d_mm = delta_u_pix * (SID/SDD) * pix_mm

    return {
        "det_center": det_center,
        "mean_center": mean_center,
        "delta_u_pix": float(delta_u_pix),
        "d_mm": float(d_mm),
        "SID": SID, "SDD": SDD, "pix_mm": pix_mm,
        "V": V, "R": R, "C": C
    }

def cleanup_keep_only_prep(prefix):
    """Behåll bara .prep. Ta bort .air/.scan/.log om de finns."""
    for ext in (".air", ".scan", ".log"):
        p = prefix + ext
        if os.path.exists(p):
            os.remove(p)

def make_ct():
    ct = xc.CatSim("Phantom_Default","Scanner_Default","Physics_Default","Protocol_Default")
    ct.load_cfg("Protocol_Default","Physics_Default","Phantom_Default","Scanner_Default","Recon_Default")
    ct.protocol.scanTypes = [1,1,1]
    return ct


# -------- main --------
if not os.path.isdir(ALLZ_DIR):
    raise FileNotFoundError(f"ALLZ_DIR finns inte: {ALLZ_DIR}")

z_dirs = list_z_dirs(ALLZ_DIR)
print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[INFO] PHANTOM_CFG   = {PHANTOM_CFG}")
print(f"[INFO] Hittade {len(z_dirs)} slices i {ALLZ_DIR}")

os.makedirs(OUTROOT, exist_ok=True)

orig_cfg_text = open(PHANTOM_CFG, "r").read()
base_offset = read_centeroffset_from_cfg(orig_cfg_text)
if base_offset is None:
    raise RuntimeError("Kunde inte läsa phantom.centerOffset från Phantom_Default.cfg")
print("[INFO] Bas centerOffset =", base_offset)

try:
    for zi, zpath in z_dirs:
        phantom_json = os.path.join(zpath, "breast_voxelized_2D.json")
        if not os.path.exists(phantom_json):
            print(f"[SKIP] saknar {phantom_json}")
            continue

        outdir = os.path.join(OUTROOT, f"z{zi:03d}")
        os.makedirs(outdir, exist_ok=True)

        # 0) återställ cfg till baseline innan slice
        open(PHANTOM_CFG, "w").write(orig_cfg_text)

        # 1) PASS1 (mät)
        ct1 = make_ct()
        ct1.phantom.callback = "Phantom_Voxelized"
        ct1.phantom.filename = phantom_json

        prefix1 = os.path.join(outdir, f"breast2d_{ENERGY_TAG}_z{zi:03d}_pass1")
        ct1.resultsName = prefix1

        print(f"\n[RUN pass1] z{zi:03d}")
        ct1.run_all()

        prep1 = ensure_prep(prefix1, ct1)
        if prep1 is None or not os.path.exists(prep1):
            print(f"[WARN] Kunde inte få .prep för pass1: {prefix1}")
            # städa ändå
            cleanup_keep_only_prep(prefix1)
            if prep1 and os.path.exists(prep1):
                os.remove(prep1)
            continue

        cfg1 = ct1.get_current_cfg()
        info = compute_dmm_from_prep(prep1, cfg1, threshold=THRESH)
        d_mm = info["d_mm"]
        du = info["delta_u_pix"]
        print(f"  delta_u_pix={du:.2f}  => d_mm={d_mm:.2f} mm")

        # PASS1: vi behöver inte spara nåt -> ta bort prep1 också
        cleanup_keep_only_prep(prefix1)
        if os.path.exists(prep1):
            os.remove(prep1)

        # 2) skriv ny centerOffset i Phantom_Default.cfg
        if not np.isfinite(d_mm):
            print("  [SKIP] Tom/svart slice (kunde inte estimera center). Skipping pass2.")
            continue

        new_offset = base_offset.copy()
        new_offset[OFFSET_INDEX] = base_offset[OFFSET_INDEX] - d_mm
        write_centeroffset_to_cfg(PHANTOM_CFG, new_offset)
        print(f"  -> phantom.centerOffset = {new_offset}")

        # 3) PASS2 (spara ENBART .prep)
        ct2 = make_ct()
        ct2.phantom.callback = "Phantom_Voxelized"
        ct2.phantom.filename = phantom_json

        # Bestäm spektrumfil beroende på namn
        if "40kVp" in ENERGY_TAG:
            ct2.protocol.spectrumFilename = "tungsten_tar7.0_120_filt.dat"
        elif "60kVp" in ENERGY_TAG:
            ct2.protocol.spectrumFilename = "tungsten_tar7.0_90_filt.dat"

        prefix2 = os.path.join(outdir, f"breast2d_{ENERGY_TAG}_z{zi:03d}")
        ct2.resultsName = prefix2

        print(f"[RUN pass2] z{zi:03d}")
        ct2.run_all()

        prep2 = ensure_prep(prefix2, ct2)
        if prep2 and os.path.exists(prep2):
            # Ta bort allt utom .prep
            cleanup_keep_only_prep(prefix2)
            print(f"  -> sparade {prep2}")
        else:
            print("  [WARN] pass2 saknar .prep (men .air/.scan kan finnas).")
            # städa så vi inte lämnar stora filer
            cleanup_keep_only_prep(prefix2)

        # 4) liten loggfil (valfri men ofta nyttig)
        with open(os.path.join(outdir, "center_offset_used.txt"), "w") as f:
            f.write(f"base_centerOffset = {base_offset}\n")
            f.write(f"measured_d_mm     = {d_mm}\n")
            f.write(f"new_centerOffset  = {new_offset}\n")
            f.write(f"threshold         = {THRESH}\n")
            f.write(f"delta_u_pix       = {info['delta_u_pix']}\n")
            f.write(f"det_center_col    = {info['det_center']}\n")
            f.write(f"mean_center_col   = {info['mean_center']}\n")
            f.write(f"SID               = {info['SID']}\n")
            f.write(f"SDD               = {info['SDD']}\n")
            f.write(f"pix_mm            = {info['pix_mm']}\n")

    print("\n[DONE] Alla slices klara! (sparade endast .prep för pass2)")

finally:
    open(PHANTOM_CFG, "w").write(orig_cfg_text)
    print("[INFO] Återställde Phantom_Default.cfg till original.")

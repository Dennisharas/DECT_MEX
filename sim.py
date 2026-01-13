import os
import warnings
import numpy as np


# --- Stryp BLAS-trådar (stabilare på macOS) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


import gecatsim as xc
from gecatsim.pyfiles import CommonTools as CT


# --- Konstanter ---
THR_P   = 1e-8
TOPK_R  = 64
EPS_SUM = 1e-12


# ---------------------------
# Hjälpfunktioner (plot, I/O)
# ---------------------------
def save_profile_png(arr_1d, out_png, title):
   import matplotlib
   matplotlib.use("Agg", force=True)
   import matplotlib.pyplot as plt
   fig = plt.figure()
   plt.plot(arr_1d)
   plt.title(title)
   plt.xlabel("Detector column")
   plt.ylabel("p = -log(I/I0)")
   plt.tight_layout()
   fig.savefig(out_png, dpi=150)
   plt.close(fig)
   print("Saved:", out_png)


def save_sinogram_png(sino_vc, out_png, c0=None, title="Sinogram"):
   import matplotlib
   matplotlib.use("Agg", force=True)
   import matplotlib.pyplot as plt
   S = sino_vc.copy()
   if np.isfinite(S).any():
       vmin = np.nanpercentile(S, 5)
       vmax = np.nanpercentile(S, 95)
       if not np.isfinite(vmin): vmin = np.nanmin(S)
       if not np.isfinite(vmax): vmax = np.nanmax(S)
       if vmin == vmax: vmax = vmin + 1.0
   else:
       vmin, vmax = 0, 1
   fig = plt.figure()
   plt.imshow(S, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
   if c0 is not None:
       plt.axvline(c0, ls='--', lw=1)
   plt.title(title)
   plt.xlabel("Detector column")
   plt.ylabel("View index")
   cbar = plt.colorbar()
   cbar.set_label("p")
   plt.tight_layout()
   fig.savefig(out_png, dpi=150)
   plt.close(fig)
   print("Saved:", out_png)


def load_phantom_mid_slice(json_path):
   import json
   with open(os.path.expanduser(json_path), "r") as f:
       J = json.load(f)
   vx = float(J.get("x_voxel_size", [1.0])[0])
   vy = float(J.get("y_voxel_size", [1.0])[0])
   vz = float(J.get("z_voxel_size", [1.0])[0])
   Xn = int(J.get("x_size", [0])[0]  or J.get("cols",   [0])[0])
   Yn = int(J.get("y_size", [0])[0]  or J.get("rows",   [0])[0])
   Zn = int(J.get("z_size", [0])[0]  or J.get("slices", [0])[0])
   vpaths = J.get("volume_path")
   if not vpaths and "volumes" in J:
       vpaths = [v["volume_path"] for v in J["volumes"]]
   vols = [np.load(os.path.expanduser(p)) for p in vpaths]


   def to_zyx(a):
       sh = np.array(a.shape)
       axes = [-1,-1,-1]
       for ax, N in enumerate((Zn, Yn, Xn)):
           axes[ax] = int(np.where(sh == N)[0][0])
       return np.transpose(a, axes)
   vols_zyx = [to_zyx(v.astype(np.float32, copy=False)) for v in vols]
   sumvol = vols_zyx[0] + vols_zyx[1] + vols_zyx[2]
   Z, Y, X = sumvol.shape
   zm = Z // 2
   sum_zmid = np.clip(sumvol[zm], 0.0, 1.0)
   return sum_zmid, (vx, vy, vz), (Z, Y, X)


def save_phantom_overlay_png(sum_zmid, vx, vy, center_offset_x_mm, dx_recommend_mm,
                            out_png, title="Phantom z-mid with iso"):
   import matplotlib
   matplotlib.use("Agg", force=True)
   import matplotlib.pyplot as plt
   Y, X = sum_zmid.shape
   x_mm = (np.arange(X) - (X-1)/2.0) * vx
   x_iso_now   = -float(center_offset_x_mm)
   x_iso_after = x_iso_now - float(dx_recommend_mm)
   fig = plt.figure(figsize=(6.5, 4.2))
   extent = [x_mm[0], x_mm[-1], 0, Y*vy]
   plt.imshow(sum_zmid, cmap="gray", origin="lower", vmin=0, vmax=1, extent=extent, aspect="equal")
   plt.axvline(x_iso_now,   ls="--", lw=1.2, label=f"iso nu (x={x_iso_now:+.1f} mm)")
   plt.axvline(x_iso_after, ls=":",  lw=1.2, label=f"iso efter Δx ({dx_recommend_mm:+.1f} mm)")
   plt.title(title)
   plt.xlabel("x (mm) i phantom"); plt.ylabel("y (mm) i phantom")
   plt.legend()
   plt.tight_layout()
   fig.savefig(out_png, dpi=150)
   plt.close(fig)
   print("Saved:", out_png)


def main():
   import matplotlib
   matplotlib.use("Agg", force=True)


   HERE = os.path.dirname(__file__)
   PKG  = os.path.abspath(os.path.join(HERE, "..", "gecatsim"))


   # Registera paths
   CT.my_path.paths = {
       "cfg"     : os.path.join(PKG, "cfg"),
       "spectrum": os.path.join(PKG, "spectrum"),
       "bowtie"  : os.path.join(PKG, "bowtie"),
       "material": os.path.join(PKG, "material"),
       "phantom" : os.path.join(PKG, "phantom"),
       "lib"     : os.path.join(PKG, "lib"),
   }


   # Skapa ct + load_cfg
   ct = xc.CatSim("Phantom_Default", "Scanner_Default", "Physics_Default", "Protocol_Default")
   try:
       ct.load_cfg("Protocol_Default", "Physics_Default", "Recon_Default")
   except Exception:
       try:
           ct.load_cfg(None, None, "Recon_Default")
       except Exception:
           pass
   cfg = ct.get_current_cfg()


   # Overrides
   ct.protocol.viewsPerRotation = 500
   ct.protocol.viewCount        = 500
   ct.protocol.startViewId      = 0
   ct.protocol.stopViewId       = ct.protocol.startViewId + ct.protocol.viewCount - 1


   ct.protocol.airViewCount     = 1
   ct.protocol.offsetViewCount  = 0
   ct.protocol.scanTypes        = [1, 0, 1]   # air, offset, phantom


   # Geometri
   ct.scanner.sid = 540.0
   ct.scanner.sdd = 950.0
   ct.scanner.detectorRowSize   = 1.0
   ct.scanner.detectorRowCount  = 800
   ct.scanner.detectorRowsPerMod= 800
   ct.scanner.detectorRowOffset = 0.0
   ct.scanner.detectorColOffset = 0.25
   ct.scanner.detectorColSize   = 1.0
   ct.scanner.targetAngle       = 7.0
   ct.scanner.focalspotLength   = 1.0


   # Fysik förenklad
   for a in ("enableQuantumNoise", "enableElectronicNoise"):
       if hasattr(cfg.physics, a): setattr(cfg.physics, a, 0)
   for a in ("viewSampleCount","colSampleCount","rowSampleCount","srcXSampleCount","srcYSampleCount"):
       if hasattr(cfg.physics, a): setattr(cfg.physics, a, 1)


   # Phantom
   ct.phantom.callback = "Phantom_Voxelized"
   ct.phantom.filename = os.path.expanduser(
       "~/Desktop/DECT_Breast_Project/xcist_voxelized_bundle_cropped3/breast_voxelized_cropped3.json"
   )
   # Hårdsätt centerOffset (från din match_center_offset.py)
   # ct.phantom.centerOffset = [-22.1, -15.7, 47.9]
   # ct.phantom.centerOffset = [- 639.1, -15.7, -327]


   # --- Carry-in av rekommenderad förflyttning från föregående körning ---
   # --- Carry-in av rekommenderad förflyttning från föregående körning ---
   import json
   state_dir   = os.path.dirname(__file__)
   carry_path  = os.path.join(state_dir, "results", "next_offset.json")
   last_path   = os.path.join(state_dir, "results", "last_offset.json")


   # Bas-offset om vi aldrig kört tidigare
   base_offset = [-485.0, -15.7, -258.0]


   # Max tillåtet steg per körning (mm)
   max_step_xy = 50.0
   max_step_z  = 50.0


   # 1) Läs "senaste center" om finns, annars använd base_offset
   if os.path.exists(last_path):
    try:
        last = json.load(open(last_path, "r"))
        current_center = list(map(float, last.get("center", base_offset)))
        print("[CARRY-IN] Using last_center from file:", current_center)
    except Exception as e:
        current_center = base_offset
        print("[CARRY-IN] Failed to read last_offset.json, using base:", current_center, "(", e, ")")
    else:
       current_center = base_offset
       print("[CARRY-IN] No last_offset.json, using base:", current_center)


   # 2) Läs rekommenderad Δ (om finns) och KLIPP stegen
    add = [0.0, 0.0, 0.0]
    if os.path.exists(carry_path):
       try:
           carry = json.load(open(carry_path, "r"))
           add_raw = carry.get("add", [0.0, 0.0, 0.0])
           dx_clip = float(np.clip(add_raw[0], -max_step_xy, max_step_xy))
           dz_clip = float(np.clip(add_raw[2], -max_step_z,  max_step_z))
           add = [dx_clip, 0.0, dz_clip]
           print(f"[CARRY-IN] Applying clipped Δoffset: {add}")
       except Exception as e:
           print("[CARRY-IN] next_offset.json unreadable, ignoring Δ (", e, ")")


    # 3) Nu ACKUMULERAR vi: ny center = förra center + Δ
    ct.phantom.centerOffset = [
        float(current_center[0] + add[0]),
        float(current_center[1] + add[1]),
        float(current_center[2] + add[2]),
    ]
    print("[FORCE] Using hard-set centerOffset:", ct.phantom.centerOffset)


   print("[FORCE] Using hard-set centerOffset:", ct.phantom.centerOffset)


   ct.protocol.bowtie = []
   if hasattr(ct.phantom, "projectorNumThreads"):
    ct.phantom.projectorNumThreads = 1


   # Spektrum
   # Spektrum (80 kVp, använd förfiltrerat spektrum och lägg INTE på extra filter)
    ct.protocol.spectrumCallback = "Spectrum"
    ct.protocol.spectrumFilename = "xcist_kVp80_tar7_bin1.dat"  # mer “rått” 80 kVp

    ct.protocol.bowtie = []
    ct.protocol.filterCallback = "Xray_Filter"
    ct.protocol.flatFilter = ['Cu', 0.2]  # t.ex. 0.2 mm Cu; prova 0.1–0.3 mm i en känslighetsanalys


   # Resultatnamn + rensa gamla för just detta resultsName
   ct.resultsName = os.path.join(HERE, "results", "sino_breast_voxelized")
   os.makedirs(os.path.dirname(ct.resultsName), exist_ok=True)
   from glob import glob
   for p in glob(ct.resultsName + ".*"):
       try:
           os.remove(p)
           print("Deleted old:", p)
       except FileNotFoundError:
           pass


   # Sanity-utskrift
   print("[pre-run] vpr =", ct.protocol.viewsPerRotation,
         "viewCount =", ct.protocol.viewCount,
         "stopViewId =", ct.protocol.stopViewId)
   print("USING PHANTOM:", ct.phantom.filename)
   print("USING OFFSET:",  ct.phantom.centerOffset)
   print("VIEWS:", ct.protocol.viewCount, "SCAN TYPES:", ct.protocol.scanTypes)
   print("DETECTOR:", getattr(ct.scanner,"detectorCallback","?"),
         "ColOffset:", getattr(ct.scanner,"detectorColOffset","?"))


   # Kör simuleringen
   ct.run_all()


   # Läs råfiler
   v = ct.protocol.viewCount
   r = ct.scanner.detectorRowCount
   c = ct.scanner.detectorColCount
   col_pitch = float(ct.scanner.detectorColSize)
   SID   = float(ct.scanner.sid)
   SDD   = float(ct.scanner.sdd)


   prep_path   = ct.resultsName + ".prep"
   scan_path   = ct.resultsName + ".scan"
   air_path    = ct.resultsName + ".air"
   offset_path = ct.resultsName + ".offset"


   # Iso-höjd (mm)
   row_pitch = float(getattr(ct.scanner, "detectorRowSize", 1.0))
   iso_height_mm = r * row_pitch * SID / SDD
   print(f"[ISO-HEIGHT] detector rows = {r}, pitch = {row_pitch:.3f} mm -> iso-height ≈ {iso_height_mm:.1f} mm")


   # Phantom Z-extent (mm)
   try:
       import json
       JJ = json.load(open(os.path.expanduser(ct.phantom.filename), "r"))
       Zn = int(JJ.get("z_size", [0])[0] or JJ.get("slices", [0])[0])
       vz = float(JJ.get("z_voxel_size", [1.0])[0])
       print(f"[PHANTOM-Z] extent ≈ {Zn*vz:.1f} mm")
   except Exception:
       pass


   # Läs rå
   air_rc   = CT.rawread(air_path,  [r, c],    "float")
   scan_vrc = CT.rawread(scan_path, [v, r, c], "float")
   if os.path.exists(offset_path):
       off_rc = CT.rawread(offset_path, [r, c], "float")
       print("Offset present: YES")
   else:
       off_rc = 0.0
       print("Offset present: NO")


   print("SHAPES → air:", air_rc.shape, "scan:", scan_vrc.shape,
         "off:", (off_rc.shape if isinstance(off_rc, np.ndarray) else "scalar"))
   assert scan_vrc.shape == (v, r, c)
   assert air_rc.shape == (r, c)


   # Sanera NaN/Inf
   air_rc   = np.nan_to_num(air_rc,   nan=0.0, posinf=0.0, neginf=0.0)
   scan_vrc = np.nan_to_num(scan_vrc, nan=0.0, posinf=0.0, neginf=0.0)


   # Diagnostik innan logg
   a_min, a_med, a_max = float(air_rc.min()), float(np.median(air_rc)), float(air_rc.max())
   s_min, s_med, s_max = float(scan_vrc.min()), float(np.median(scan_vrc)), float(scan_vrc.max())
   print(f"Air   min/median/max: {a_min:.3g} / {a_med:.3g} / {a_max:.3g}")
   print(f"Scan  min/median/max: {s_min:.3g} / {s_med:.3g} / {s_max:.3g}")


   # Robust kolumnmask (air-svag)
   air_minus = air_rc - off_rc
   pos_mask = air_minus > 0
   thr = max(1e-3, np.quantile(air_minus[pos_mask], 0.005)) if np.any(pos_mask) else 1e-3
   bad_cols = (air_minus < thr).mean(axis=0) > 0.20
   print("Bad columns (robust):", int(bad_cols.sum()), f"av {c}")


   # -------------------------------
   # --- Robust transmission + p ---
   # -------------------------------
   den_eps = max(1e-3, 0.01 * a_med)
   denom   = np.maximum(air_rc - off_rc, den_eps)      # (r,c)
   num     = np.maximum(scan_vrc - off_rc, 0.0)        # (v,r,c)


   T = num / denom[None, :, :]                         # (v,r,c)
   T = np.clip(T, 1e-6, 1.5)


   p_vrc = -np.log(T).astype(np.float32)               # (v,r,c)


   # Dämpningsgrad för rad-urval
   D_vrc = 1.0 - T                                     # (v,r,c)
   D_vr  = np.nanmean(D_vrc, axis=2)                   # (v,r)
   cov_r = D_vr.mean(axis=0)                           # (r,)


   thr_cov = max(0.002, 0.10 * (np.nanmax(cov_r) + EPS_SUM))
   rows_mask_raw = cov_r > thr_cov


   use_rows = rows_mask_raw.copy()
   if np.count_nonzero(use_rows) < max(8, min(TOPK_R, len(cov_r)//4)):
       K = min(TOPK_R, len(cov_r))
       idx = np.argsort(cov_r)[-K:]
       use_rows[:] = False
       use_rows[idx] = True


   # Radmedel över valda rader → (v,c)
   p_for_center = np.nanmean(p_vrc[:, use_rows, :], axis=1)


   # Diagnostik p
   try:
       p5, p50, p95 = np.nanpercentile(p_vrc, [5, 50, 95])
       print(f"[P STATS] p5={p5:.4g}, p50={p50:.4g}, p95={p95:.4g}")
   except Exception:
       pass


   # --- Row-averaged sinogram vs angle (0..360°) ---
   S_vc = np.nanmean(p_vrc[:, use_rows, :], axis=1)  # (v,c)


   # Vinkel per vy (antag jämn sampling över 360°)
   views = int(ct.protocol.viewCount)
   deg_per_view = 360.0 / float(ct.protocol.viewsPerRotation)  # 360/360 = 1.0 i din setup
   angles = np.arange(views, dtype=np.float32) * deg_per_view  # [0, 1, 2, ..., 359] deg


   # För "closed loop": lägg till första kolumnen på slutet (0°=360°)
   S_closed = np.concatenate([S_vc, S_vc[0:1, :]], axis=0)   # (v+1, c)
   ang_closed = np.concatenate([angles, [360.0]], axis=0)    # (v+1,)


   # Plotta mot grader (x=0..360)
   import matplotlib
   matplotlib.use("Agg", force=True)
   import matplotlib.pyplot as plt


   fig = plt.figure()
   # extent = [xmin, xmax, ymin, ymax]; vi vill ha x i grader, y = kolumnindex
   plt.imshow(S_closed.T,  # (c, v+1) för att få kolumner som y, vy/grad som x
          extent=[float(ang_closed[0]), float(ang_closed[-1]), 0, S_closed.shape[1]],
          aspect='auto', origin='lower',
          vmin=np.nanpercentile(S_vc, 5), vmax=np.nanpercentile(S_vc, 95))
   plt.xlabel("Gantry angle (deg)")
   plt.ylabel("Detector column")
   plt.title("Row-averaged sinogram vs angle (closed at 360°)")
   cbar = plt.colorbar(); cbar.set_label("p = -ln(I/I0)")
   out_dir = os.path.dirname(prep_path)   # använd globala 'os' från filens topp
   plt.tight_layout()
   plt.savefig(os.path.join(out_dir, "sinogram_rowavg_vs_angle.png"), dpi=150)
   plt.close(fig)
   print("Saved:", os.path.join(out_dir, "sinogram_rowavg_vs_angle.png"))




   # --------------------------
   # Centrering (X och Z-led)
   # --------------------------
   c0   = (c - 1) / 2.0 + float(getattr(ct.scanner, "detectorColOffset", 0.0))
   cols = np.arange(c, dtype=np.float32)


   W     = np.where(np.isfinite(p_for_center) & (p_for_center > THR_P), p_for_center, 0.0)
   den_w = W.sum(axis=1) + EPS_SUM
   xc_v  = (W * cols[None, :]).sum(axis=1) / den_w   # centroid per vy


   xc_mean = float(np.nanmean(xc_v))
   dcol    = xc_mean - c0
   dx_mm   = dcol * col_pitch * SID / SDD
   print(f"[CENTER] (row-avg) mean col-centroid = {xc_mean:.2f}, mid = {c0:.2f}, Δcol = {dcol:+.2f}")
   print(f"[CENTER] Rekommenderad justering: phantom.centerOffset[0] += {dx_mm:+.1f} mm")


   # Z-led: rad-centroid
   rows = np.arange(r, dtype=np.float32)
   Wr = np.where(np.isfinite(p_vrc) & (p_vrc > THR_P), p_vrc, 0.0).sum(axis=2)  # (v,r)
   r0 = (r - 1) / 2.0 + float(getattr(ct.scanner, "detectorRowOffset", 0.0))
   zc_v = (Wr * rows[None, :]).sum(axis=1) / (Wr.sum(axis=1) + EPS_SUM)
   zc_mean = float(np.nanmean(zc_v))
   drow = zc_mean - r0
   dz_mm = drow * row_pitch * SID / SDD
   print(f"[CENTER-Z] mean row-centroid = {zc_mean:.2f}, mid = {r0:.2f}, Δrow = {drow:+.2f}")
   print(f"[CENTER-Z] Rekommenderad justering: phantom.centerOffset[2] += {dz_mm:+.1f} mm")


   # --------------------------
   # Coverage / plots / PREP
   # --------------------------
   n_rows_raw = int(np.count_nonzero(rows_mask_raw))
   height_mm_iso_raw = n_rows_raw * row_pitch * SID / SDD
   print(f"[FOV-Y] RAW rows above thr: {n_rows_raw}/{r} ({100*n_rows_raw/max(1,r):.1f}%), ≈ {height_mm_iso_raw:.1f} mm vid iso")


   coverage_ratio = n_rows_raw / max(1, r)
   low_cov = coverage_ratio < 0.25
   if low_cov:
       print(f"[SAFEGUARD] Vertical coverage {coverage_ratio*100:.1f}% < 25% → IGNORE Δx/Δy this run.")
       dx_mm = 0.0
       # (vi låter dz_mm stå för rapport/info men för carry-out tar vi bara med dx om >25%)


   # Bästa rad att plotta
   cov_r_for_best = Wr.mean(axis=0)
   best_row = int(np.argmax(cov_r_for_best))
   print(f"[PLOT] Best row to visualize: {best_row} / {r-1}")


   # Maskera dåliga kolumner för visning
   p_plot = p_vrc.copy()
   p_plot[:, :, bad_cols] = np.nan
   p_plot = np.clip(p_plot, -5, 10)


   out_dir = os.path.dirname(prep_path)
   save_sinogram_png(p_plot[:, best_row, :], os.path.join(out_dir, "sinogram_bestrow.png"),
                     c0=c0, title=f"Sinogram (best row={best_row})")
   save_sinogram_png(np.nanmean(p_plot[:, use_rows, :], axis=1),
                     os.path.join(out_dir, "sinogram_rowavg.png"), c0=c0, title="Sinogram (row-averaged)")
   save_sinogram_png(p_plot[:, r//2, :],
                     os.path.join(out_dir, "sinogram_midrow_robust.png"), c0=c0,
                     title=f"Sinogram (mid-row) — V={v}")


   # Upptagen bredd (kolumner) på radmedel
   vis_cols = np.any(np.isfinite(p_for_center) & (p_for_center > THR_P), axis=0)
   width_cols = int(vis_cols.sum()); frac_cols = float(vis_cols.mean())
   width_mm_iso = width_cols * col_pitch * SID / SDD
   print(f"[FOV] Upptagen bredd (row-avg) ≈ {width_cols} kolumner ({frac_cols*100:.1f}%) ≈ {width_mm_iso:.1f} mm vid iso")


   # Spara PREP (p)
   p_vrc.astype(np.float32, copy=False).tofile(prep_path)
   print("Wrote robust PREP:", prep_path)


   # Profiler
   save_profile_png(p_plot[-1, r//2, :], os.path.join(out_dir, "profile_lastview_midrow.png"),
                    "Prep profile (middle row) — last view")
   prep_last = CT.rawread(prep_path, [v, r, c], "float")[-1]
   save_profile_png(prep_last[r//2, :], os.path.join(out_dir, "prep_lastview_midrow.png"),
                    "Prep profile (middle row) — last view [reloaded]")


   # Phantom-overlay (valfritt)
   try:
       sum_zmid, (vx, vy, vz), (Z, Y, X) = load_phantom_mid_slice(ct.phantom.filename)
       save_phantom_overlay_png(sum_zmid, vx, vy,
           center_offset_x_mm=float(ct.phantom.centerOffset[0]),
           dx_recommend_mm=dx_mm,
           out_png=os.path.join(out_dir, "phantom_zmid_iso_overlay.png"),
           title="Phantom z-mid with current iso and suggested Δx")
   except Exception as e:
       print("(info) Skipping phantom overlay plot:", e)


   # Rapport
   report_txt = os.path.join(out_dir, "centering_report.txt")
   with open(report_txt, "w") as f:
       f.write(f"views={v}, rows={r}, cols={c}\n")
       f.write(f"Air   min/median/max: {a_min:.6g} / {a_med:.6g} / {a_max:.6g}\n")
       f.write(f"Scan  min/median/max: {s_min:.6g} / {s_med:.6g} / {s_max:.6g}\n")
       f.write(f"Bad columns: {int(bad_cols.sum())} / {c}\n")
       f.write(f"Centroid mean col: {xc_mean:.3f}  mid: {c0:.3f}  Δcol: {dcol:+.3f}\n")
       f.write(f"Recommended Δx for phantom.centerOffset[0]: {dx_mm:+.3f} mm\n")
       f.write(f"Recommended Δz for phantom.centerOffset[2]: {dz_mm:+.3f} mm\n")
       f.write(f"FOV mid-row width: {width_cols} cols ({frac_cols*100:.1f}%), ≈ {width_mm_iso:.2f} mm\n")
   print("Saved:", report_txt)


   # Carry-out för nästa körning (bara om coverage OK)
   try:
       import json
       carry_path = os.path.join(HERE, "results", "next_offset.json")
       if not low_cov:
            # Klipp rekommenderad Δ innan vi sparar
           dx_clip = float(np.clip(dx_mm, -max_step_xy, max_step_xy))
           dz_clip = float(np.clip(dz_mm, -max_step_z,  max_step_z))
           add = [dx_clip, 0.0, dz_clip]
           with open(carry_path, "w") as f:
               json.dump({"add": add}, f)
           print(f"[NEXT RUN] Will add Δoffset {add} on next run.")
       else:
           if os.path.exists(carry_path):
               os.remove(carry_path)
           print("[NEXT RUN] Skipping Δoffset carry-out due to low vertical coverage.")
   except Exception:
       pass


if __name__ == "__main__":
   import multiprocessing as mp
   try:
       mp.set_start_method("spawn")
   except RuntimeError:
       pass
   warnings.filterwarnings("ignore", message="overflow encountered")
   warnings.filterwarnings("ignore", message="invalid value encountered")
   np.seterr(over='ignore', invalid='ignore')
   main()



#!/usr/bin/env python3
import os, json, warnings
import numpy as np


# Stryp trådar (stabilitet på macOS)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


import gecatsim as xc
from gecatsim.pyfiles import CommonTools as CT


# ----------- Konstanter / paths -----------
HERE         = os.path.dirname(__file__)
RESULTS_DIR  = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


PHANTOM_JSON = os.path.expanduser(
   "~/Desktop/DECT_Breast_Project/xcist_voxelized_bundle_cropped3/breast_voxelized_cropped3.json"
)


# Basoffset om vi aldrig kört förr
BASE_OFFSET  = [-485.0, -15.7, -258.0]
# Max steg per körning (mm)
MAX_STEP_XY  = 50.0
MAX_STEP_Z   = 50.0


# ---------- Hjälp: läs/skriv state ----------
def read_json(path, default):
   try:
       if os.path.exists(path):
           return json.load(open(path, "r"))
   except Exception:
       pass
   return default


def write_json(path, obj):
   with open(path, "w") as f:
       json.dump(obj, f)


# ---------- Bäring till centrerings-mått ----------
def compute_center_shifts(ct, prep_path, scan_path, air_path, offset_path):
   """Returnerar (dx_mm, dz_mm) från centroid-metoden."""
   # Läs rå
   v = int(ct.protocol.viewCount)
   r = int(ct.scanner.detectorRowCount)
   c = int(ct.scanner.detectorColCount)
   SID = float(ct.scanner.sid); SDD = float(ct.scanner.sdd)


   air_rc   = CT.rawread(air_path,  [r, c],    "float")
   scan_vrc = CT.rawread(scan_path, [v, r, c], "float")
   off_rc   = CT.rawread(offset_path, [r, c], "float") if os.path.exists(offset_path) else 0.0


   # Sanera
   air_rc   = np.nan_to_num(air_rc,   nan=0.0, posinf=0.0, neginf=0.0)
   scan_vrc = np.nan_to_num(scan_vrc, nan=0.0, posinf=0.0, neginf=0.0)


   # Transmission + p
   a_med = float(np.median(air_rc))
   den_eps = max(1e-3, 0.01 * a_med)
   denom = np.maximum(air_rc - off_rc, den_eps)        # (r,c)
   num   = np.maximum(scan_vrc - off_rc, 0.0)          # (v,r,c)
   T = num / denom[None, :, :]
   T = np.clip(T, 1e-6, 1.5)
   p_vrc = -np.log(T).astype(np.float32)


   # Välj rader med signal
   D_vrc = 1.0 - T
   D_vr  = np.nanmean(D_vrc, axis=2)                   # (v,r)
   cov_r = D_vr.mean(axis=0)                           # (r,)
   thr_cov = max(0.002, 0.10 * (np.nanmax(cov_r) + 1e-12))
   use_rows = cov_r > thr_cov
   if np.count_nonzero(use_rows) < 32:                 # fallback
       idx = np.argsort(cov_r)[-64:]
       use_rows[:] = False
       use_rows[idx] = True


   # Row-averaged sinogram (v,c)
   p_vc = np.nanmean(p_vrc[:, use_rows, :], axis=1)


   # X-centroid
   col_pitch = float(ct.scanner.detectorColSize)
   c0   = (c - 1)/2.0 + float(getattr(ct.scanner, "detectorColOffset", 0.0))
   cols = np.arange(c, dtype=np.float32)
   W    = np.where(np.isfinite(p_vc) & (p_vc > 1e-8), p_vc, 0.0)
   den  = W.sum(axis=1) + 1e-12
   xc_v = (W * cols[None, :]).sum(axis=1) / den
   xc_mean = float(np.nanmean(xc_v))
   dcol    = xc_mean - c0
   dx_mm   = dcol * col_pitch * SID / SDD


   # Z-centroid (radmedel viktat per vy)
   row_pitch = float(getattr(ct.scanner, "detectorRowSize", 1.0))
   rows = np.arange(r, dtype=np.float32)
   Wr = np.where(np.isfinite(p_vrc) & (p_vrc > 1e-8), p_vrc, 0.0).sum(axis=2)  # (v,r)
   r0 = (r - 1)/2.0 + float(getattr(ct.scanner, "detectorRowOffset", 0.0))
   zc_v = (Wr * rows[None, :]).sum(axis=1) / (Wr.sum(axis=1) + 1e-12)
   zc_mean = float(np.nanmean(zc_v))
   drow = zc_mean - r0
   dz_mm = drow * row_pitch * SID / SDD


   # Lite logg likt din tidigare
   print(f"[CENTER] (row-avg) mean col-centroid = {xc_mean:.2f}, mid = {c0:.2f}, Δcol = {dcol:+.2f}")
   print(f"[CENTER] Rekommenderad justering: phantom.centerOffset[0] += {dx_mm:+.1f} mm")
   print(f"[CENTER-Z] mean row-centroid = {zc_mean:.2f}, mid = {r0:.2f}, Δrow = {drow:+.2f}")
   print(f"[CENTER-Z] Rekommenderad justering: phantom.centerOffset[2] += {dz_mm:+.1f} mm")


   # Spara robust PREP (kan vara bra framåt)
   p_vrc.astype(np.float32, copy=False).tofile(prep_path)
   print("Wrote robust PREP:", prep_path)


   return dx_mm, dz_mm




if __name__ == "__main__":
   # På macOS hjälper detta ofta mot segfault med native libbar
   os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
   try:
       import multiprocessing as mp
       mp.set_start_method("spawn")
   except Exception:
       pass
   main()












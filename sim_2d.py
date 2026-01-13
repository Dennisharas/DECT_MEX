import os
import matplotlib.pyplot as plt
import gecatsim as xc
from gecatsim.pyfiles import CommonTools as CT

HERE = os.path.dirname(__file__)
PKG  = os.path.abspath(os.path.join(HERE, "..", "gecatsim"))

# 1) Registrera sökvägar så CatSim hittar allt (inkl. din C/C++-lib)
CT.my_path.paths = {
    "cfg"     : os.path.join(PKG, "cfg"),       # använd paketets *_Default.cfg
    "spectrum": os.path.join(PKG, "spectrum"),
    "bowtie"  : os.path.join(PKG, "bowtie"),
    "material": os.path.join(PKG, "material"),
    "phantom" : os.path.join(PKG, "phantom"),
    "lib"     : os.path.join(PKG, "lib"),
}

# 2) Skapa CatSim-objektet (detta är “ct”)
ct = xc.CatSim("Phantom_Default", "Scanner_Default", "Physics_Default", "Protocol_Default")

# ➕ LÄGG TILL DENNA RAD så Recon-params finns och .prep kan skapas:
ct.load_cfg("Protocol_Default", "Physics_Default", "Phantom_Default", "Scanner_Default", "Recon_Default")

# (valfritt) – peka på logofantomen för en snabb test
ct.phantom.callback = "Phantom_Voxelized"
ct.phantom.filename = os.path.expanduser(
    "~/Desktop/DECT_Breast_Project/xcist_voxelized_bundle_auto2d_calc/UncompressedBreast1_z202/breast_voxelized_2D.json"
)

energy_tag = "40kVp_Cu0.2"  # byt till t.ex. "120kVp_Cu0.2" i andra körningen
# Kör en phantom-scan; ct.run_all() kör sedan preprocess så .prep skapas

out_prefix = os.path.join(HERE, "results", f"breast2d_{energy_tag}")
os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

ct.protocol.scanTypes = [1, 1, 1]   # air, offset, phantom
ct.resultsName = out_prefix
os.makedirs(os.path.dirname(ct.resultsName), exist_ok=True)

# (fix för spectra-namnet på din repo – välj en som finns)
# ct.protocol.spectrumFilename = "tungsten_tar7.0_120_filt.dat"

#ct.protocol.spectrumCallback = "Spectrum"
# ct.protocol.spectrumFilename = "xcist_kVp140_tar7_bin1.dat"  # mer “rått” 80 kVp

#ct.protocol.bowtie = []
#ct.protocol.filterCallback = "Xray_Filter"
#ct.protocol.flatFilter = ['Cu', 0.2]  # t.ex. 0.2 mm Cu; prova 0.1–0.3 mm i en känslighetsanalys

# Be CatSim ladda Recon-default (vissa builds skriver då .prep automatiskt)
try:
    ct.load_cfg("Protocol_Default", "Physics_Default", "Recon_Default")
except Exception:
    # En del versioner kräver exakt tre argument – om ovan krånglar, prova:
    try:
        ct.load_cfg(None, None, "Recon_Default")
    except Exception:
        pass  # hoppa vidare till B)

# 3) Kör hela kedjan
ct.run_all()

import numpy as np, os

v = ct.protocol.viewCount
r = ct.scanner.detectorRowCount
c = ct.scanner.detectorColCount

prep_path = ct.resultsName + ".prep"
scan_path = ct.resultsName + ".scan"
air_path  = ct.resultsName + ".air"

if not os.path.exists(prep_path):
    # Läs råfiler enligt dimensionsordningen i README:
    # .air:   [col, row]
    # .scan:  [col, row, view]
    air_cr   = CT.rawread(air_path,  [c, r],       "float")        # shape (c, r)
    scan_crv = CT.rawread(scan_path, [c, r, v],    "float")        # shape (c, r, v)

    # Reordna till [v, r, c] för bekväm broadcasting
    air_rc   = air_cr.T                                           # -> (r, c)
    scan_vrc = np.transpose(scan_crv, (2, 1, 0))                  # (v, r, c)

    # p = -log( (scan - D) / (air - D) ), här saknas offset => D=0
    eps = 1e-6
    p_vrc = -np.log( np.clip(scan_vrc, eps, None) / np.clip(air_rc, eps, None) )

    # Spara som float32 binärfil så resten av verktygen kan läsa den
    p_vrc.astype(np.float32, copy=False).tofile(prep_path)
    print("Wrote:", prep_path)
else:
    print("Found existing .prep:", prep_path)

# Nu kan du läsa sista vyn och plotta
prep_last = CT.rawread(prep_path, [v, r, c], "float")[-1]
# import matplotlib.pyplot as plt
plt.plot(prep_last[r // 2, :])
plt.title("Prep profile (middle row)")
plt.xlabel("Detector column")
plt.ylabel("Signal")
plt.show()

import os, numpy as np, matplotlib.pyplot as plt
import gecatsim as xc
from gecatsim.pyfiles import CommonTools as CT

# --- prefix till dina filer ---
energy_tag = "40kVp_Cu0.2"   # ändra till exakt samma sträng som i körningen
prefix = os.path.join("results", f"breast2d_{energy_tag}")

# --- hämta dimensioner från cfg ---
ct  = xc.CatSim()                  # bara för att få current cfg
cfg = ct.get_current_cfg()
V = int(cfg.protocol.viewCount)
R = int(cfg.scanner.detectorRowCount)
C = int(cfg.scanner.detectorColCount)

# --- fall A: .prep finns redan -> läs och plott­a direkt ---
prep_path = prefix + ".prep"
if os.path.exists(prep_path):
    prep = CT.rawread(prep_path, [V, R, C], "float")  # (V,R,C)
    sino = prep[:, R//2, :]                           # mitt-raden  -> (V,C)

# --- fall B: ingen .prep -> bygg p = -log(scan/air) i RAM ---
else:
    air  = CT.rawread(prefix + ".air",  [C, R], "float").T     # (R,C)
    scan = CT.rawread(prefix + ".scan", [C, R, V], "float")    # (C,R,V)
    scan = np.transpose(scan, (2, 1, 0))                      # -> (V,R,C)

    eps  = 1e-6
    p = -np.log(np.clip(scan, eps, None) / np.clip(air, eps, None))  # (V,R,C)
    sino = p[:, R//2, :]                                            # (V,C)

# --- plot ---
plt.figure(figsize=(10,6))
plt.imshow(
    sino, cmap="gray", aspect="auto", origin="lower",
    extent=[0, C-1, 0, 360.0]    # visar vy-axeln som grader (en varv)
)
plt.colorbar(label="p = -log(I/I₀)")
plt.xlabel("Detektorkolumn")
plt.ylabel("Vinkel (view)")
plt.title("Sinogram (mitt-rad)")
plt.show()


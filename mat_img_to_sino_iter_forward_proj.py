import os
import numpy as np
import gecatsim as xc
from gecatsim.pyfiles import CommonTools as CT

# --- Sökvägar och inställningar ---
# Anpassa dessa efter din katalogstruktur
NPY_DIR = os.path.expanduser("~/LPD_MMDNet_data/training_data")
OUT_NPY_DIR = os.path.expanduser("~/DECT_Breast_Project/material_sinograms_npy")
os.makedirs(OUT_NPY_DIR, exist_ok=True)

MATERIALS = ["Adipose", "Fibroglandular", "Calcification"]

# Konfigurera CatSim-sökvägar (enligt sim_2d.py)[cite: 6]
HERE = os.path.dirname(__file__) if '__file__' in locals() else "."
PKG = os.path.abspath(os.path.join(HERE, "..", "main", "gecatsim"))

CT.my_path.paths = {
    "cfg"     : os.path.join(PKG, "cfg"),
    "spectrum": os.path.join(PKG, "spectrum"),
    "bowtie"  : os.path.join(PKG, "bowtie"),
    "material": os.path.join(PKG, "material"),
    "phantom" : os.path.join(PKG, "phantom"),
    "lib"     : os.path.join(PKG, "lib"),
}

def setup_catsim():
    ct = xc.CatSim("Phantom_Default", "Scanner_Default", "Physics_Default", "Protocol_Default")
    ct.load_cfg("Protocol_Default", "Physics_Default", "Phantom_Default", "Scanner_Default", "Recon_Default")
    ct.protocol.scanTypes = [1, 1, 1]  # air, offset, phantom
    return ct

material_mapping = {
    "Adipose": "ICRU_adipose_adult2",
    "Fibroglandular": "ICRU_breast_adult2",
    "Calcification": "ICRU_skeleton_cortical_bone_adult"
}

density_mapping = {
    "Adipose": 0.95,
    "Fibroglandular": 1.02,
    "Calcification": 1.92
}

# --- Huvudloop för att generera materialsinogram ---
print("[INFO] Startar generering av materialsinogram till .npy...")

# --- Iterera genom material och dess z-slices ---
for mat in MATERIALS:
    npy_path = os.path.join(NPY_DIR, f"Phantom_{mat}.npy")  # Justera namnet efter dina filer
    if not os.path.exists(npy_path):
        print(f"[WARN] Hittar inte volymfilen: {npy_path}")
        continue

    print(f"[INFO] Laddar 3D-volym för {mat}...")
    volume_3d = np.load(npy_path)  # Förväntad shape: (1000, 512, 512)
    num_slices = volume_3d.shape[0]

    # Inuti din loop för varje material (mat):
    xcist_mat_name = material_mapping[mat]
    xcist_density = density_mapping[mat]

    for zi in range(num_slices):
        # Plocka ut enskild 2D-slice
        slice_2d = volume_3d[zi, :, :].astype(np.float32)

        # Valfritt: Hoppa över helt tomma slices för att spara tid
        if np.max(slice_2d) == 0:
            continue

        slice_out_dir = os.path.join(OUT_NPY_DIR, f"z{zi:03d}")
        os.makedirs(slice_out_dir, exist_ok=True)

        # Spara undan Slicen som binärfil för CatSim
        bin_filename = os.path.join(slice_out_dir, f"phantom_{mat}.bin")
        slice_2d.tofile(bin_filename)

        # 2. Skapa en liten JSON-konfig som CatSim kräver för Phantom_Voxelized
        json_filename = os.path.join(slice_out_dir, f"phantom_{mat}.json")
        json_content = f"""{{
            "fname": "{os.path.basename(bin_filename)}",
            "mat_name": ["{xcist_mat_name}"],
            "mat_density": [0.0012, {xcist_density}],
            "n_materials": 1,
            "voxelSize": [0.5, 0.5, 0.5],
            "centerOffset": [0.0, 0.0, 0.0]
        }}"""
        with open(json_filename, "w") as f:
            f.write(json_content)

        # Initiera och kör CatSim
        ct = setup_catsim()
        ct.phantom.callback = "Phantom_Voxelized"
        ct.phantom.filename = json_filename
        
        out_prefix = os.path.join(slice_out_dir, f"sino_{mat}")
        ct.resultsName = out_prefix
        
        try:
            ct.run_all()
        except Exception as e:
            print(f"[ERROR] Fel vid slice {zi}, material {mat}: {e}")
            continue

        # Hämta .prep-filen (eller beräkna om den saknas)
        prep_path = out_prefix + ".prep"
        air_path = out_prefix + ".air"
        scan_path = out_prefix + ".scan"

        v = int(ct.protocol.viewCount)
        r = int(ct.scanner.detectorRowCount)
        c = int(ct.scanner.detectorColCount)

        if os.path.exists(prep_path):
            prep = CT.rawread(prep_path, [v, r, c], "float")
        else:
            air_cr = CT.rawread(air_path, [c, r], "float")
            scan_crv = CT.rawread(scan_path, [c, r, v], "float")
            air_rc = air_cr.T
            scan_vrc = np.transpose(scan_crv, (2, 1, 0))
            eps = 1e-6
            prep = -np.log(np.clip(scan_vrc, eps, None) / np.clip(air_rc, eps, None))

        # Extrahera 2D-sinogrammet (mitt-raden) -> Shape: (Views, Detectors)
        material_sino_2d = prep[:, r // 2, :]

        # Spara som .npy för träning
        final_npy_save_path = os.path.join(slice_out_dir, f"target_sino_{mat}.npy")
        np.save(final_npy_save_path, material_sino_2d)

        print(f"[DONE] Sparade sinogram för {mat}, slice {zi}")

print("[INFO] Alla materialsinogram har genererats från 3D-volymerna!")
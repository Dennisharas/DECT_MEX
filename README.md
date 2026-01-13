# DECT Breast Project

This repository contains the code developed for a master thesis project on  
**dual-energy CT (DECT) breast imaging**, with a focus on simulation-based
material decomposition and quantitative image evaluation.

The project implements a full research pipeline, from breast DICOM data to
simulated DECT images, material decomposition, and quantitative analysis.
The code is intended for academic and research use.

---

## Project overview

The main goals of this project are to:

- Convert breast DICOM data into XCIST-compatible voxelized phantoms
- Simulate dual-energy CT acquisitions using XCIST
- Reconstruct images using FDK
- Perform material decomposition into breast-relevant materials
- Quantitatively evaluate reconstruction and decomposition quality

The project is simulation-driven and is designed to support methodological
evaluation rather than clinical deployment.

---

## Repository structure

DECT_Breast_Project/
├── README.md
├── scripts/
│ ├── dicom_to_phantom_allz.py
│ ├── sim_run_all.py
│ ├── sim_slim_run_all.py
│ ├── sim.py
│ ├── xcist_fdk_from_prep.py
│ ├── image_matdecomp_3mat_from_fdk.py
│ ├── mat_decomp_Iter.py
│ └── run_eval_mmdstyle2.py
└── docs/ 


All executable scripts are located in the `scripts/` directory.  
Documentation and extended instructions are located in `docs/`.

---

## Pipeline summary

The processing pipeline consists of the following main stages:

1. **DICOM to phantom conversion**  
   A 3D breast DICOM volume is converted into slice-wise 2D voxelized phantoms
   compatible with XCIST.

2. **DECT simulation**  
   Dual-energy CT simulations (e.g. low- and high-energy spectra) are performed
   using XCIST.

3. **Image reconstruction**  
   Simulated projection data are reconstructed into image volumes using
   filtered backprojection (FDK).

4. **Material decomposition**  
   Reconstructed images are decomposed into material-specific components
   (e.g. adipose, glandular tissue, skin, calcification-related material).

5. **Quantitative evaluation**  
   Reconstruction and material decomposition results are evaluated using
   quantitative metrics such as RMSE, CNR, and SSIM.

Each stage is implemented as one or more standalone scripts.

---

## Script overview

### `dicom_to_phantom_allz.py`
Converts a breast DICOM series into slice-wise 2D XCIST voxelized phantoms.

Main functionality:
- Loads and sorts a DICOM series
- Builds a 3D breast mask (air removed)
- Generates material fraction maps
- Adds spatially consistent synthetic calcifications
- Saves XCIST-compatible `.bin` and `.json` phantom files
- Generates PNG visualizations for verification

---

### `sim_run_all.py`
Runs full dual-energy XCIST simulations for all phantom slices.

Typically used for final, full-resolution simulations.

---

### `sim_slim_run_all.py`
A lightweight version of `sim_run_all.py`.

Used for faster testing, debugging, and parameter tuning.

---

### `sim.py`
Contains lower-level simulation configuration and shared utilities
used by the simulation run scripts.

---

### `xcist_fdk_from_prep.py`
Performs FDK reconstruction from XCIST `.prep` files.

Produces reconstructed image volumes used for subsequent material
decomposition.

---

### `image_matdecomp_3mat_from_fdk.py`
Performs three-material decomposition on reconstructed FDK volumes.

Used as a baseline or reference material decomposition method.

---

### `mat_decomp_Iter.py`
Implements iterative material decomposition.

Used to refine material separation and compare iterative and
non-iterative approaches.

---

### `run_eval_mmdstyle2.py`
Performs quantitative evaluation of reconstruction and material
decomposition results.

Metrics include:
- Root mean square error (RMSE)
- Contrast-to-noise ratio (CNR)
- Structural similarity index (SSIM)

---

## Dependencies

Typical dependencies include:
- Python 3.9+
- numpy
- scipy
- matplotlib
- pydicom
- XCIST (GE CatSim)

Some scripts require a local XCIST installation and compatible
configuration files.

---

## Data and privacy

- DICOM data and generated phantom volumes are **not included** in this repository.
- Large binary files (`.dcm`, `.bin`, `.npy`) are intentionally excluded.
- Paths to local data directories are configured inside the scripts.

---

## Usage documentation

Detailed, step-by-step instructions on how to run each script and the
complete pipeline are provided in a separate document in the `docs/`
directory.

---

## Author

Master thesis project  
Author: **Daniela Wallis Cullblom**

---

## License

This repository is intended for academic and research use.

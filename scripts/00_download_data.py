"""
Downloading fMRI data and atlas.
"""
import os
from nilearn import datasets

dataDir = os.path.abspath('./data')
os.makedirs(dataDir, exist_ok=True)


# --- Downloading development fMRI dataset ---
print("Downloading development fMRI dataset...")
developmentData = datasets.fetch_development_fmri(
    n_subjects=30,
    data_dir=dataDir,
    verbose=1
)
print(f"\nDownloaded {len(developmentData.func)} subjects")


# --- Downloading Schaefer 100 parcel atlas with 7 networks ---
print("\nDownloading Schaefer 100 parcel atlas with 7 networks...")
schaefer = datasets.fetch_atlas_schaefer_2018(
    n_rois=100,
    yeo_networks=7,
    resolution_mm=2,
    data_dir=dataDir
)
print(f"Atlas: {schaefer.maps}")
print(f"  Number of ROIs: {len(schaefer.labels)}")


# --- Downloading MSDL atlas as backup ---
print("\nDownloading MSDL atlas as backup since it has nice 3D coords.")
msdl = datasets.fetch_atlas_msdl(data_dir=dataDir)
print(f"MSDL atlas with {len(msdl.labels)} regions")

print("\nAll downloads complete.")

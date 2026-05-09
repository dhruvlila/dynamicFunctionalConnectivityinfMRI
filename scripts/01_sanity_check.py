"""
Running quick sanity check to plot a brain and extract one time series.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, maskers

dataDir = os.path.abspath('./data')


# --- Loading data ---
print("Loading data...")
devData = datasets.fetch_development_fmri(n_subjects=1, data_dir=dataDir)
schaefer = datasets.fetch_atlas_schaefer_2018(
    n_rois=100, yeo_networks=7, data_dir=dataDir
)


# --- Plotting the atlas on a brain ---
print("Plotting atlas...")
fig = plotting.plot_roi(
    schaefer.maps,
    title="Schaefer 100 Atlas 7 Networks",
    display_mode='ortho'
)
fig.savefig('./figures/atlas_check.png', dpi=150, bbox_inches='tight')
plt.close()


# --- Extracting time series from the first subject ---
print("Extracting time series...")
masker = maskers.NiftiLabelsMasker(
    labels_img=schaefer.maps,
    standardize='zscore_sample',
    memory='nilearn_cache',
    verbose=1
)

timeSeries = masker.fit_transform(devData.func[0])
print(f"\nTime series shape: {timeSeries.shape}")
print(f"  ({timeSeries.shape[0]} timepoints x {timeSeries.shape[1]} ROIs)")


# --- Plotting the time series of first 5 ROIs ---
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(5):
    ax.plot(timeSeries[:, i], label=f'ROI {i+1}', alpha=0.7)
ax.set_xlabel('Time TR')
ax.set_ylabel('BOLD z scored')
ax.set_title('Sample BOLD Time Series')
ax.legend()
fig.tight_layout()
fig.savefig('./figures/sample_timeseries.png', dpi=150)
plt.close()


print("\nSanity check complete.")

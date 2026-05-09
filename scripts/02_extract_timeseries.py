"""
Extracting ROI time series for all subjects.
Saving to ./results/timeseries.npz for reuse.
"""

import os
import numpy as np
import pandas as pd
from nilearn import datasets, maskers
from tqdm import tqdm

dataDir = os.path.abspath('./data')
resultsDir = os.path.abspath('./results')
os.makedirs(resultsDir, exist_ok=True)


# --- Loading dataset and atlas ---
print("Loading dataset metadata...")
nSubjects = 30
devData = datasets.fetch_development_fmri(
    n_subjects=nSubjects, data_dir=dataDir
)
schaefer = datasets.fetch_atlas_schaefer_2018(
    n_rois=100, yeo_networks=7, resolution_mm=2, data_dir=dataDir
)


# --- Converting phenotypic info to DataFrame for safe indexing ---
phenotypic = pd.DataFrame(devData.phenotypic).reset_index(drop=True)
print(f"\nPhenotypic columns: {list(phenotypic.columns)}")
print(f"Phenotypic shape: {phenotypic.shape}")

print("\nDataset info:")
print(f"  N subjects: {len(devData.func)}")
if 'Child_Adult' in phenotypic.columns:
    print(f"  Children:   {sum(phenotypic['Child_Adult'] == 'child')}")
    print(f"  Adults:     {sum(phenotypic['Child_Adult'] == 'adult')}")


def parseNetwork(label):
    """
    Parsing atlas labels into network names.
    """
    parts = label.split('_')
    return parts[2] if len(parts) >= 3 else 'Unknown'


# --- Parsing atlas labels into network names ---
labels = [l.decode('utf-8') if isinstance(l, bytes) else l 
          for l in schaefer.labels]

networks = [parseNetwork(l) for l in labels]
uniqueNetworks = sorted(set(networks))
print(f"\nNetworks found: {uniqueNetworks}")
print("ROI counts per network:")
for net in uniqueNetworks:
    print(f"  {net:10s}: {networks.count(net)}")


# --- Extracting time series for each subject ---
print("\nExtracting time series for all subjects...")

masker = maskers.NiftiLabelsMasker(
    labels_img=schaefer.maps,
    standardize='zscore_sample',
    standardize_confounds=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0,
    memory='nilearn_cache',
    memory_level=1,
    verbose=0
)

allTimeseries = []
subjectInfo = []

for i in tqdm(range(nSubjects), desc="Subjects"):
    funcFile = devData.func[i]
    confounds = devData.confounds[i] if devData.confounds[i] else None
    
    ts = masker.fit_transform(funcFile, confounds=confounds)
    allTimeseries.append(ts)
    
    info = {'subject_idx': i, 'n_timepoints': ts.shape[0]}
    for col in ['Age', 'Child_Adult', 'Gender']:
        if col in phenotypic.columns:
            try:
                info[col.lower()] = phenotypic[col].iloc[i]
            except Exception:
                info[col.lower()] = None
    subjectInfo.append(info)


# --- Saving everything ---
nTimepoints = [ts.shape[0] for ts in allTimeseries]
print(f"\nTimepoints per subject: min={min(nTimepoints)}, max={max(nTimepoints)}")

minT = min(nTimepoints)
allTimeseriesTrimmed = np.stack([ts[:minT] for ts in allTimeseries])
print(f"Trimmed shape: {allTimeseriesTrimmed.shape}")
print(f"  ({nSubjects} subjects, {minT} timepoints, {len(labels)} ROIs)")

subjectDf = pd.DataFrame(subjectInfo)
subjectDf.to_csv(f'{resultsDir}/subject_info.csv', index=False)

np.savez(
    f'{resultsDir}/timeseries.npz',
    timeseries=allTimeseriesTrimmed,
    networks=np.array(networks),
    labels=np.array(labels),
    n_subjects=nSubjects,
    n_timepoints=minT,
    n_rois=len(labels),
    tr=2.0
)

print(f"\nSaved to {resultsDir}/timeseries.npz")
print(f"Saved subject info to {resultsDir}/subject_info.csv")


# --- Showing subject info summary ---
print("\nSubject info preview:")
print(subjectDf.head(10))

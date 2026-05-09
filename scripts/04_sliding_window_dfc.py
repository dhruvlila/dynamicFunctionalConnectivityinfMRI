"""
Sliding window dynamic FC.
Saves windowed FC matrices for all subjects.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

resultsDir = os.path.abspath('./results')
figDir = os.path.abspath('./figures')


# --- Setting up parameters ---
windowLength = 30
stepSize = 1
useTapered = True


# --- Loading data ---
print("Loading sorted time series...")
sd = np.load(f'{resultsDir}/static_fc.npz', allow_pickle=True)
timeseriesSorted = sd['timeseries_sorted'] if 'timeseries_sorted' in sd else sd['timeseriesSorted']
networksSorted = sd['networks_sorted'] if 'networks_sorted' in sd else sd['networksSorted']
networkBoundaries = sd['network_boundaries'] if 'network_boundaries' in sd else sd['networkBoundaries']
nSubj, nTime, nRois = timeseriesSorted.shape

print(f"Data: {nSubj} subjects, {nTime} timepoints, {nRois} ROIs")
print(f"Window: {windowLength} TRs, step={stepSize}")


# --- Computing sliding window FC ---
nWindows = (nTime - windowLength) // stepSize + 1
print(f"Will compute {nWindows} windows per subject")

if useTapered:
    taper = np.hamming(windowLength)
else:
    taper = np.ones(windowLength)

windowFc = np.zeros((nSubj, nWindows, nRois, nRois), dtype=np.float32)
windowStarts = np.arange(0, nTime - windowLength + 1, stepSize)

print("\nComputing sliding window FC for all subjects...")
for s in tqdm(range(nSubj), desc="Subjects"):
    ts = timeseriesSorted[s]
    for w, start in enumerate(windowStarts):
        chunk = ts[start:start + windowLength]
        chunkT = chunk * taper[:, None]
        chunkT = chunkT - chunkT.mean(axis=0)
        
        std = chunkT.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        chunkZ = chunkT / std
        cov = (chunkZ.T @ chunkZ) / (windowLength - 1)
        
        np.fill_diagonal(cov, 1.0)
        windowFc[s, w] = np.clip(cov, -1, 1)

print(f"Window FC tensor shape: {windowFc.shape}")


# --- Computing FC variance map ---
print("Computing FC variance map...")
fcVariance = windowFc.var(axis=1).mean(axis=0)
np.fill_diagonal(fcVariance, 0)


# --- Saving windowed FC tensor ---
print("\nSaving windowed FC tensor...")
np.savez_compressed(
    f'{resultsDir}/window_fc.npz',
    windowFc=windowFc,
    windowStarts=windowStarts,
    windowLength=windowLength,
    stepSize=stepSize,
    nWindows=nWindows,
    fcVariance=fcVariance
)
print(f"Saved {resultsDir}/window_fc.npz")

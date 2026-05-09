"""
Running surrogate data null test for non stationarity.

Generating N phase randomized surrogates per subject. Phase randomization
preserves the power spectrum AND static cross correlations,
but destroys any non linear and non stationary temporal structure.

If real data shows greater windowed FC variability than surrogates,
that is direct evidence for non stationarity.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

resultsDir = os.path.abspath('./results')
figDir = os.path.abspath('./figures')

nSurrogates = 100
seed = 42


# --- Loading real data ---
print("Loading data...")
sd = np.load(f'{resultsDir}/static_fc.npz', allow_pickle=True)
ts = sd['timeseriesSorted'] if 'timeseriesSorted' in sd else sd['timeseries_sorted']
networksSorted = sd['networksSorted'] if 'networksSorted' in sd else sd['networks_sorted']
networkBoundaries = sd['networkBoundaries'] if 'networkBoundaries' in sd else sd['network_boundaries']

wd = np.load(f'{resultsDir}/window_fc.npz')
realWindowFc = wd['windowFc'] if 'windowFc' in wd else wd['window_fc']
windowLength = int(wd['windowLength'] if 'windowLength' in wd else wd['window_length'])

nSubj, nTime, nRois = ts.shape
nWindows = realWindowFc.shape[1]
triuIdx = np.triu_indices(nRois, k=1)
nEdges = len(triuIdx[0])
taper = np.hamming(windowLength)
print(f"Data: {nSubj} subj, T={nTime}, ROIs={nRois}, "
      f"windows={nWindows}, edges={nEdges}")


def phaseRandomizeMultivariate(timeseries, rng):
    """
    Applying multivariate phase randomization.
    
    Applying same random phase shift to all ROIs at each frequency.
    Preserving the auto spectrum and cross spectrum while destroying non stationary structure.
    """
    tTime, rRois = timeseries.shape
    fftData = np.fft.rfft(timeseries, axis=0)
    nFreqs = fftData.shape[0]
    
    randomPhases = rng.uniform(0, 2*np.pi, size=nFreqs)
    randomPhases[0] = 0
    if tTime % 2 == 0:
        randomPhases[-1] = 0
    
    phaseShift = np.exp(1j * randomPhases)[:, None]
    fftSurr = fftData * phaseShift
    surr = np.fft.irfft(fftSurr, n=tTime, axis=0)
    return surr.astype(np.float32)


def computeDfcEdgeVariance(timeseries, windowLength, taper):
    """
    Computing variance of each FC edge across sliding windows.
    """
    tTime, rRois = timeseries.shape
    nWin = tTime - windowLength + 1
    triuLocal = np.triu_indices(rRois, k=1)
    edgeTs = np.zeros((nWin, len(triuLocal[0])), dtype=np.float32)
    
    for w in range(nWin):
        chunk = timeseries[w:w+windowLength] * taper[:, None]
        chunk = chunk - chunk.mean(axis=0)
        std = chunk.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        chunk = chunk / std
        cov = (chunk.T @ chunk) / (windowLength - 1)
        edgeTs[w] = cov[triuLocal]
    
    return edgeTs.var(axis=0)


# --- Computing real dFC edge variance ---
print("\nComputing real data dFC edge variance...")
realVar = np.zeros((nSubj, nEdges), dtype=np.float32)
for s in range(nSubj):
    realVar[s] = computeDfcEdgeVariance(ts[s], windowLength, taper)
print(f"  Real var matrix: {realVar.shape}")


# --- Generating surrogates and computing their dFC edge variance ---
print(f"\nComputing surrogate dFC variance for {nSurrogates} per subject...")

rng = np.random.default_rng(seed)
surrVar = np.zeros((nSubj, nSurrogates, nEdges), dtype=np.float32)

for s in tqdm(range(nSubj), desc="Subjects"):
    for k in range(nSurrogates):
        surrogate = phaseRandomizeMultivariate(ts[s], rng)
        surrVar[s, k] = computeDfcEdgeVariance(
            surrogate, windowLength, taper)

print(f"  Surrogate var tensor: {surrVar.shape}")


# --- Running statistical comparison ---
print("\nRunning statistical comparison...")

realEdgeVarGroup = realVar.mean(axis=0)
surrEdgeVarGroup = surrVar.mean(axis=0)

pPerEdge = (surrEdgeVarGroup >= realEdgeVarGroup[None, :]).mean(axis=0)

realTotalVar = realEdgeVarGroup.sum()
surrTotalVar = surrEdgeVarGroup.sum(axis=1)
pGlobal = (surrTotalVar >= realTotalVar).mean()
zGlobal = (realTotalVar - surrTotalVar.mean()) / surrTotalVar.std()

print("\nGLOBAL STATIONARITY TEST")
print(f"  Real total dFC variance:        {realTotalVar:.3f}")
print(f"  Surrogate variance mean SD: "
      f"{surrTotalVar.mean():.3f} and {surrTotalVar.std():.3f}")
print(f"  Z score:                        {zGlobal:+.2f}")
print(f"  p value one sided:            {pGlobal:.4f}")

if zGlobal > 0:
    print("  Real data shows MORE FC variability than expected")
    print("  under stationarity hence evidence for non stationarity.")
else:
    print("  Real data does NOT exceed stationary surrogates.")
    print("  Observed dynamics are consistent with stationarity.")

realTotalPerSubj = realVar.sum(axis=1)
surrTotalPerSubj = surrVar.sum(axis=2)
zPerSubj = (realTotalPerSubj - surrTotalPerSubj.mean(axis=1)) / \
              surrTotalPerSubj.std(axis=1)
pPerSubj = (surrTotalPerSubj >= realTotalPerSubj[:, None]).mean(axis=1)

nSig = (pPerSubj < 0.05).sum()
print("\nPER SUBJECT TEST")
print(f"  Subjects with p < 0.05: {nSig} out of {nSubj} "
      f"({100*nSig/nSubj:.0f}%)")
print(f"  Mean Z score across subjects: {zPerSubj.mean():+.2f}")


# --- Saving results ---
np.savez(
    f'{resultsDir}/surrogate_test.npz',
    realVar=realVar, surrVar=surrVar,
    realEdgeVarGroup=realEdgeVarGroup,
    surrEdgeVarGroup=surrEdgeVarGroup,
    pPerEdge=pPerEdge,
    pPerSubj=pPerSubj, zPerSubj=zPerSubj,
    realTotalVar=realTotalVar, surrTotalVar=surrTotalVar,
    pGlobal=pGlobal, zGlobal=zGlobal,
    nSurrogates=nSurrogates
)
print(f"\nSaved {resultsDir}/surrogate_test.npz")

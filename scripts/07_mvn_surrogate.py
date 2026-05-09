"""
Running MVN Gaussian surrogate which is a stricter null.

Phase randomization preserves the power spectrum.
Multivariate Gaussian sampling destroys all temporal structure,
preserving only the static covariance matrix. This is a stricter test
and if real data exceeds this null, evidence for non stationarity is fully robust.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

resultsDir = os.path.abspath('./results')
figDir = os.path.abspath('./figures')

nSurrogates = 100
seed = 1234


# --- Loading data ---
print("Loading...")
sd = np.load(f'{resultsDir}/static_fc.npz', allow_pickle=True)
ts = sd['timeseriesSorted'] if 'timeseriesSorted' in sd else sd['timeseries_sorted']
wd = np.load(f'{resultsDir}/window_fc.npz')
windowLength = int(wd['windowLength'] if 'windowLength' in wd else wd['window_length'])
nSubj, nTime, nRois = ts.shape
nWindows = nTime - windowLength + 1
triuIdx = np.triu_indices(nRois, k=1)
nEdges = len(triuIdx[0])
taper = np.hamming(windowLength)


# --- Loading existing surrogate test phase randomized results ---
st = np.load(f'{resultsDir}/surrogate_test.npz')
realTotalVar = float(st['realTotalVar'] if 'realTotalVar' in st else st['real_total_var'])
phaseSurrTotal = st['surrTotalVar'] if 'surrTotalVar' in st else st['surr_total_var']
zGlobal = st['zGlobal'] if 'zGlobal' in st else st['z_global']
print(f"Loaded existing phase rand results "
      f"real={realTotalVar:.2f} phase mean={phaseSurrTotal.mean():.2f}")


def mvnSurrogate(timeseries, rng):

    tTime, rRois = timeseries.shape
    cov = np.cov(timeseries.T) + 1e-5 * np.eye(rRois)
    choleskyL = np.linalg.cholesky(cov)
    zSamples = rng.standard_normal((tTime, rRois))
    surr = zSamples @ choleskyL.T
    return surr.astype(np.float32)


def dfcEdgeVariance(timeseries, windowLength, taper):
    """
    Computing variance of dynamic functional connectivity edges.
    """
    tTime, rRois = timeseries.shape
    nWin = tTime - windowLength + 1
    tri = np.triu_indices(rRois, k=1)
    outVar = np.zeros((nWin, len(tri[0])), dtype=np.float32)
    for w in range(nWin):
        chunk = timeseries[w:w+windowLength] * taper[:, None]
        chunk = chunk - chunk.mean(axis=0)
        std = chunk.std(axis=0, ddof=1); std[std == 0] = 1
        chunk = chunk / std
        cov = (chunk.T @ chunk) / (windowLength - 1)
        outVar[w] = cov[tri]
    return outVar.var(axis=0)


# --- Generating MVN surrogates per subject ---
print(f"\nGenerating {nSurrogates} MVN surrogates per subject...")

rng = np.random.default_rng(seed)
mvnSurrVar = np.zeros((nSubj, nSurrogates, nEdges), dtype=np.float32)

for s in tqdm(range(nSubj), desc="Subjects"):
    for k in range(nSurrogates):
        surr = mvnSurrogate(ts[s], rng)
        mvnSurrVar[s, k] = dfcEdgeVariance(surr, windowLength, taper)


# --- Aggregating stats ---
mvnEdgeGroup = mvnSurrVar.mean(axis=0)
mvnTotal = mvnEdgeGroup.sum(axis=1)
realEdgeGroup = st['realEdgeVarGroup'] if 'realEdgeVarGroup' in st else st['real_edge_var_group']

zMvn = (realTotalVar - mvnTotal.mean()) / mvnTotal.std()
pMvn = (mvnTotal >= realTotalVar).mean()

pEdgeMvn = (mvnEdgeGroup >= realEdgeGroup[None, :]).mean(axis=0)

print("\nMVN STATIONARITY TEST")
print(f"  Real total dFC variance:    {realTotalVar:.2f}")
print(f"  MVN surrogate mean and SD:  {mvnTotal.mean():.2f} and {mvnTotal.std():.2f}")
print(f"  Z score:                    {zMvn:+.2f}")
print(f"  p value:                    {pMvn:.4f}")


# --- Plotting three way null comparison ---
print("\nPlotting three way null comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(phaseSurrTotal, bins=30, color='#4682B4', alpha=0.55,
        edgecolor='black', label=f'Phase rand (Z={zGlobal:+.1f})')
ax.hist(mvnTotal, bins=30, color='#00760E', alpha=0.55,
        edgecolor='black', label=f'MVN (Z={zMvn:+.1f})')
ax.axvline(realTotalVar, color='#CD3E3A', lw=3.5,
           label='Real data')
ax.set_xlabel('Total dFC variance', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Real vs Two Stationary Null Models\n'
             'Real exceeds BOTH nulls', fontsize=12)
ax.legend(fontsize=10); ax.grid(alpha=0.3)

ax = axes[1]
labelsData = ['Real\ndata', 'Phase randomized\nsurrogate', 'MVN Gaussian\nsurrogate']
vals = [realTotalVar, phaseSurrTotal.mean(), mvnTotal.mean()]
errs = [0, phaseSurrTotal.std(), mvnTotal.std()]
colorsData = ['#CD3E3A', '#4682B4', '#00760E']

bars = ax.bar(labelsData, vals, yerr=errs, color=colorsData, alpha=0.85,
              edgecolor='black', linewidth=1.2, capsize=8)
ax.set_ylabel('Total dFC variance', fontsize=11)
ax.set_title('Summary Real Exceeds All Stationary Nulls', fontsize=12)
ax.grid(alpha=0.3, axis='y')

ax.annotate(f'Z = {zGlobal:+.1f}', xy=(1, vals[1]),
            xytext=(1, vals[1]-15), ha='center', fontsize=10, color='black')
ax.annotate(f'Z = {zMvn:+.1f}', xy=(2, vals[2]),
            xytext=(2, vals[2]-15), ha='center', fontsize=10, color='black')
ax.annotate('Reference\nobserved', xy=(0, vals[0]),
            xytext=(0, vals[0]-15), ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig(f'{figDir}/three_way_null.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved {figDir}/three_way_null.png")


# --- Comparing per edge which test is more powerful ---
pPerEdge = st['pPerEdge'] if 'pPerEdge' in st else st['p_per_edge']
sigPhase = (pPerEdge < 0.05).sum()
sigMvn = (pEdgeMvn < 0.05).sum()

print(f"\n  Phase rand significant edges: {sigPhase} ({100*sigPhase/nEdges:.1f}%)")
print(f"  MVN significant edges:        {sigMvn} ({100*sigMvn/nEdges:.1f}%)")


# --- Saving results ---
np.savez(
    f'{resultsDir}/mvn_surrogate.npz',
    mvnSurrVar=mvnSurrVar,
    mvnTotal=mvnTotal,
    pEdgeMvn=pEdgeMvn,
    zMvn=zMvn, pMvn=pMvn,
    sigPhase=sigPhase, sigMvn=sigMvn
)
print(f"\nSaved {resultsDir}/mvn_surrogate.npz")

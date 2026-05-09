"""
Analyzing static functional connectivity and handling background labels.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind

resultsDir = os.path.abspath('./results')
figDir = os.path.abspath('./figures')


# --- Loading time series and handling background label ---
print("Loading time series...")
data = np.load(f'{resultsDir}/timeseries.npz', allow_pickle=True)
timeseries = data['timeseries']
networks = list(data['networks'])
labels = list(data['labels'])
nSubj, nTime, nRois = timeseries.shape


# --- Dropping background label to align ---
if len(networks) == nRois + 1:
    print(f"Dropping Background label (atlas had {len(networks)} labels, "
          f"data has {nRois} ROIs)")
    networks = networks[1:]
    labels = labels[1:]

networks = np.array(networks)
labels = np.array(labels)
print(f"Aligned: {nRois} ROIs, {len(networks)} network labels")
print(f"Shape: {timeseries.shape}")


# --- Sorting ROIs by network for block diagonal display ---
networkOrder = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn',
                 'Limbic', 'Cont', 'Default']
networkColors = {
    'Vis':         '#781286', 'SomMot':      '#4682B4',
    'DorsAttn':    '#00760E', 'SalVentAttn': '#C43AFA',
    'Limbic':      '#DCF8A4', 'Cont':        '#E69422',
    'Default':     '#CD3E3A',
}
networkFull = {
    'Vis': 'Visual', 'SomMot': 'Somatomotor',
    'DorsAttn': 'Dorsal Attn', 'SalVentAttn': 'Salience/VAN',
    'Limbic': 'Limbic', 'Cont': 'Frontoparietal',
    'Default': 'Default Mode'
}

sortIdx = []
networkBoundaries = [0]
for net in networkOrder:
    netIdx = np.where(networks == net)[0]
    sortIdx.extend(netIdx)
    networkBoundaries.append(len(sortIdx))
sortIdx = np.array(sortIdx)
networksSorted = networks[sortIdx]
nRoisUsed = len(sortIdx)
print(f"Using {nRoisUsed} ROIs across {len(networkOrder)} networks")

timeseriesSorted = timeseries[:, :, sortIdx]


# --- Computing static FC ---
print("\nComputing static FC...")
fcMatrices = np.zeros((nSubj, nRoisUsed, nRoisUsed))
for s in range(nSubj):
    fcMatrices[s] = np.corrcoef(timeseriesSorted[s].T)

fcZ = np.arctanh(np.clip(fcMatrices, -0.999, 0.999))
groupFc = np.tanh(np.mean(fcZ, axis=0))
np.fill_diagonal(groupFc, 0)


# --- Plotting group average FC matrix ---
print("Plotting group FC matrix...")
fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(groupFc, cmap='RdBu_r', vmin=-0.6, vmax=0.6,
               interpolation='nearest')

for b in networkBoundaries[1:-1]:
    ax.axhline(b - 0.5, color='black', lw=1.2)
    ax.axvline(b - 0.5, color='black', lw=1.2)

for i, net in enumerate(networkOrder):
    start, end = networkBoundaries[i], networkBoundaries[i+1]
    mid = (start + end) / 2
    ax.add_patch(Rectangle((-3.5, start - 0.5), 2, end - start,
                           facecolor=networkColors[net], clip_on=False))
    ax.add_patch(Rectangle((start - 0.5, -3.5), end - start, 2,
                           facecolor=networkColors[net], clip_on=False))
    ax.text(-5, mid, networkFull[net], rotation=0, ha='right',
            va='center', fontsize=10, fontweight='bold',
            color=networkColors[net])

ax.set_xlim(-3.5, nRoisUsed - 0.5)
ax.set_ylim(nRoisUsed - 0.5, -3.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'Group-Average Static Functional Connectivity\n'
             f'(N={nSubj}, Schaefer atlas, organized by 7 Yeo networks)',
             fontsize=13, pad=15)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Pearson correlation', fontsize=11)
plt.tight_layout()
plt.savefig(f'{figDir}/group_static_fc.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved to {figDir}/group_static_fc.png")


# --- Computing network level connectivity ---
print("Computing network-level FC...")
nNets = len(networkOrder)
networkFc = np.zeros((nNets, nNets))
for i, netI in enumerate(networkOrder):
    idxI = np.where(networksSorted == netI)[0]
    for j, netJ in enumerate(networkOrder):
        idxJ = np.where(networksSorted == netJ)[0]
        if i == j:
            sub = groupFc[np.ix_(idxI, idxJ)]
            mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
            networkFc[i, j] = sub[mask].mean()
        else:
            networkFc[i, j] = groupFc[np.ix_(idxI, idxJ)].mean()


# --- Analyzing within vs between network distribution ---
withinVals, betweenVals = [], []
for i in range(nRoisUsed):
    for j in range(i+1, nRoisUsed):
        if networksSorted[i] == networksSorted[j]:
            withinVals.append(groupFc[i, j])
        else:
            betweenVals.append(groupFc[i, j])

t, p = ttest_ind(withinVals, betweenVals)

print(f"\n  Within mean:  {np.mean(withinVals):.4f}")
print(f"  Between mean: {np.mean(betweenVals):.4f}")
print(f"  t = {t:.2f}, p = {p:.2e}")


# --- Saving processed data ---
np.savez(
    f'{resultsDir}/static_fc.npz',
    fcMatrices=fcMatrices,
    groupFc=groupFc,
    networkFc=networkFc,
    networksSorted=networksSorted,
    sortIdx=sortIdx,
    networkBoundaries=np.array(networkBoundaries),
    timeseriesSorted=timeseriesSorted,
    networkOrder=np.array(networkOrder, dtype=object)
)
print(f"\nSaved {resultsDir}/static_fc.npz")

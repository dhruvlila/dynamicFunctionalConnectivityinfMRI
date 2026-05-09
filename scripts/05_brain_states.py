"""
Identifying recurring brain states via k means clustering on windowed FC matrices.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.patches import Rectangle
import seaborn as sns

resultsDir = os.path.abspath('./results')
figDir = os.path.abspath('./figures')


# --- Loading windowed FC ---
print("Loading windowed FC...")
wd = np.load(f'{resultsDir}/window_fc.npz')
windowFc = wd['windowFc'] if 'windowFc' in wd else wd['window_fc']
nSubj, nWindows, nRois, _ = windowFc.shape

sd = np.load(f'{resultsDir}/static_fc.npz', allow_pickle=True)
networksSorted = sd['networksSorted'] if 'networksSorted' in sd else sd['networks_sorted']
networkBoundaries = sd['networkBoundaries'] if 'networkBoundaries' in sd else sd['network_boundaries']
networkOrder = list(sd['networkOrder'] if 'networkOrder' in sd else sd['network_order'])

print(f"Window FC shape: {windowFc.shape}")


# --- Vectorizing upper triangle of each FC matrix ---
print("Vectorizing FC matrices upper triangle...")
triuIdx = np.triu_indices(nRois, k=1)
nEdges = len(triuIdx[0])
print(f"  Edges per FC matrix: {nEdges}")

fcVectors = windowFc[:, :, triuIdx[0], triuIdx[1]]
fcFlat = fcVectors.reshape(-1, nEdges)
print(f"  Total FC vectors for clustering: {fcFlat.shape}")


# --- Finding optimal k via elbow and silhouette ---
print("\nFinding optimal number of states k is 2 to 6...")
ks = range(2, 7)
inertias, silhouettes = [], []

np.random.seed(42)
subIdx = np.random.choice(len(fcFlat), 
                           size=min(3000, len(fcFlat)), 
                           replace=False)

for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=300)
    labelsK = km.fit_predict(fcFlat)
    inertias.append(km.inertia_)
    sil = silhouette_score(fcFlat[subIdx], labelsK[subIdx])
    silhouettes.append(sil)
    print(f"  k={k} inertia={km.inertia_:.0f} silhouette={sil:.4f}")


# --- Plotting elbow and silhouette ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(ks, inertias, 'o-', lw=2, ms=9, color='#4682B4')
axes[0].set_xlabel('Number of clusters k'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method'); axes[0].grid(alpha=0.3)

axes[1].plot(ks, silhouettes, 'o-', lw=2, ms=9, color='#CD3E3A')
axes[1].set_xlabel('Number of clusters k'); axes[1].set_ylabel('Silhouette score')
axes[1].set_title('Silhouette Analysis'); axes[1].grid(alpha=0.3)
bestK = ks[int(np.argmax(silhouettes))]
axes[1].axvline(bestK, color='gray', linestyle='--', alpha=0.7,
                label=f'Best k = {bestK}')
axes[1].legend()
plt.suptitle('Choosing the Number of Brain States', fontsize=13)
plt.tight_layout()
plt.savefig(f'{figDir}/choose_k.png', dpi=180, bbox_inches='tight')
plt.close()
print(f"Saved {figDir}/choose_k.png")
print(f"Optimal k by silhouette: {bestK}")


# --- Fitting final k means with optimal k ---
finalK = max(bestK, 4)
print(f"\nFitting final k means with k = {finalK}...")
kmeans = KMeans(n_clusters=finalK, n_init=50, random_state=42, max_iter=500)
stateLabels = kmeans.fit_predict(fcFlat)
stateLabels = stateLabels.reshape(nSubj, nWindows)


# --- Recovering centroids as full matrices ---
centroids = np.zeros((finalK, nRois, nRois))
for k in range(finalK):
    mMatrix = np.zeros((nRois, nRois))
    mMatrix[triuIdx] = kmeans.cluster_centers_[k]
    mMatrix = mMatrix + mMatrix.T
    centroids[k] = mMatrix


# --- Sorting states by occupancy ---
counts = np.array([np.sum(stateLabels == k) for k in range(finalK)])
order = np.argsort(-counts)
remap = {old: new for new, old in enumerate(order)}
stateLabels = np.vectorize(remap.get)(stateLabels)
centroids = centroids[order]
counts = counts[order]
fractions = counts / counts.sum()

print("\nState occupancy:")
for k in range(finalK):
    print(f"  State {k+1}: {fractions[k]*100:5.1f}% "
          f"({counts[k]} of {counts.sum()} windows)")


# --- Computing network level summary of each state ---
print("Computing network level summaries...")
nNets = len(networkOrder)
stateNetworkFc = np.zeros((finalK, nNets, nNets))
for k in range(finalK):
    for i, ni in enumerate(networkOrder):
        idxI = np.where(networksSorted == ni)[0]
        for j, nj in enumerate(networkOrder):
            idxJ = np.where(networksSorted == nj)[0]
            sub = centroids[k][np.ix_(idxI, idxJ)]
            if i == j:
                mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
                stateNetworkFc[k, i, j] = sub[mask].mean()
            else:
                stateNetworkFc[k, i, j] = sub.mean()


# --- Saving results ---
np.savez(
    f'{resultsDir}/brain_states.npz',
    centroids=centroids,
    stateLabels=stateLabels,
    stateNetworkFc=stateNetworkFc,
    fractions=fractions,
    finalK=finalK, bestK=bestK,
    inertias=np.array(inertias),
    silhouettes=np.array(silhouettes)
)
print(f"\nSaved {resultsDir}/brain_states.npz")

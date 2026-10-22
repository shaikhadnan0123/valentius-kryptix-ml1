"""Unsupervised learning pipeline: K-Means + PCA on the Wine dataset (labels hidden)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

RANDOM_STATE = 42

# Always write outputs next to this script, creating the figures folder if missing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load data; keep true labels ONLY for an after-the-fact sanity check
wine = load_wine(as_frame=True)
X = wine.data
true_labels = wine.target  # never used for fitting
print("Shape:", X.shape)

# 2. Scale
X_scaled = StandardScaler().fit_transform(X)

# 3. Choose k
ks = range(2, 11)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, km.labels_))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sils[-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertias, "o-"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(list(ks), sils, "o-", color="tab:orange"); ax[1].set(title="Silhouette score", xlabel="k", ylabel="Mean silhouette")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "elbow_silhouette.png"), dpi=150); plt.close()

best_k = list(ks)[int(np.argmax(sils))]
print("Best k by silhouette:", best_k)

# 4. Final K-Means
km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
clusters = km.labels_

# 5. PCA
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
evr = pca.explained_variance_ratio_
print("Explained variance ratio:", evr, "total:", evr.sum())

# 6. Plot
plt.figure(figsize=(7, 5.5))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=45, edgecolor="k", linewidth=.3)
cent = pca.transform(km.cluster_centers_)
plt.scatter(cent[:, 0], cent[:, 1], c="red", marker="X", s=200, edgecolor="k", label="Centroids")
plt.xlabel(f"PC1 ({evr[0]:.1%} variance)"); plt.ylabel(f"PC2 ({evr[1]:.1%} variance)")
plt.title(f"K-Means (k={best_k}) clusters in PCA space — {evr.sum():.1%} variance captured")
plt.legend(*sc.legend_elements(), title="Cluster", loc="best"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "pca_clusters.png"), dpi=150); plt.close()

# 7. Interpretation helpers
df = X.copy(); df["cluster"] = clusters
profile = df.groupby("cluster").mean().round(2)
print("\nCluster sizes:\n", df["cluster"].value_counts().sort_index())
print("\nCluster means (original units):\n", profile.T)
profile.T.to_csv(os.path.join(BASE_DIR, "cluster_profiles.csv"))

z = pd.DataFrame(X_scaled, columns=X.columns); z["cluster"] = clusters
zprof = z.groupby("cluster").mean().round(2)
print("\nCluster means (z-scores):\n", zprof.T)

loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=["PC1", "PC2"]).round(2)
print("\nPCA loadings:\n", loadings.sort_values("PC1"))

print("\nExamples per cluster:")
for c in range(best_k):
    print(f"\n-- Cluster {c} --"); print(df[df.cluster == c].head(3).round(2).to_string())

print("\nSanity check only — ARI vs hidden cultivars:", round(adjusted_rand_score(true_labels, clusters), 3))
print("Cross-tab:\n", pd.crosstab(clusters, true_labels.map(dict(enumerate(wine.target_names))), rownames=["cluster"]))
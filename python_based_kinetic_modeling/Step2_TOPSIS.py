import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================================================
# 1. LOAD DATA
# =========================================================
data = pd.read_excel(r"e:\Document 2026\Hydrogen project\Dark\Biohydrogen from CVH\data_\Step2_TOPSIS.xlsx")

samples = data.iloc[:, 0].astype(str).values

X = data[['Hmax', 'CO2max', 'H2/CO2', 'Reducing sugar', 'pH']].values.astype(float)

# =========================================================
# 2. BENEFIT / COST SETTING
# =========================================================
benefit_cols = [0, 1, 2, 3]   # all benefit except pH
cost_cols = [4]

# =========================================================
# 3. NORMALIZATION
# =========================================================
norm = X / np.sqrt((X ** 2).sum(axis=0))

# =========================================================
# 4. WEIGHTING
# =========================================================
weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
weighted = norm * weights

# =========================================================
# 5. IDEAL SOLUTIONS
# =========================================================
ideal_best = np.zeros(weighted.shape[1])
ideal_worst = np.zeros(weighted.shape[1])

for j in range(weighted.shape[1]):
    if j in benefit_cols:
        ideal_best[j] = np.max(weighted[:, j])
        ideal_worst[j] = np.min(weighted[:, j])
    else:
        ideal_best[j] = np.min(weighted[:, j])
        ideal_worst[j] = np.max(weighted[:, j])

# =========================================================
# 6. DISTANCES
# =========================================================
d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

# =========================================================
# 7. TOPSIS SCORE
# =========================================================
score = d_worst / (d_best + d_worst)

# =========================================================
# 8. RESULT TABLE
# =========================================================
result = pd.DataFrame({
    "Sample": samples,
    "Hmax": X[:, 0],
    "CO2max": X[:, 1],
    "H2/CO2": X[:, 2],
    "Reducing Sugar": X[:, 3],
    "pH": X[:, 4],
    "TOPSIS_score": score
})

result["Rank"] = result["TOPSIS_score"].rank(ascending=False, method="dense")
result = result.sort_values("TOPSIS_score", ascending=False)

print(result)

# =========================================================
# 9. EXPORT
# =========================================================
output_file = r"e:\Document 2026\Hydrogen project\Dark\Biohydrogen from CVH\result_\Step2_TOPSIS.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    result.to_excel(writer, index=False)

# =========================================================
# 10. TOPSIS RANKING 
# =========================================================
plot_data = result.copy()
plot_data["Sample"] = plot_data["Sample"].astype(str)

colors = cm.viridis(np.linspace(0.9, 0.2, len(plot_data)))

plt.figure(figsize=(8,6))

plt.bar(plot_data["Sample"], plot_data["TOPSIS_score"], color=colors)

for i, v in enumerate(plot_data["TOPSIS_score"]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=12, rotation=45)

plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)

plt.ylabel("TOPSIS Score", fontsize=14)
plt.title("TOPSIS Ranking with Performance Degradation Trend", fontsize=14)

plt.ylim(0, max(plot_data["TOPSIS_score"]) + 0.1)

plt.tight_layout()
plt.show()

# =========================================================
# 11. HEATMAP (SAMPLE vs VARIABLE PERFORMANCE + VALUES)
# =========================================================

plt.figure(figsize=(8,6))

# gunakan data asli
heat_data = pd.DataFrame(X, columns=['Hmax','CO$_2$max','H$_2$/CO$_2$','Reducing Sugar','pH'])
heat_data.insert(0, "Sample", samples)
heat_data = heat_data.set_index("Sample")

# normalisasi per kolom
heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())

sns.heatmap(
    heat_norm,
    cmap="YlOrRd",
    linewidths=0.3,
    linecolor="white",
    annot=True,                 # 🔥 INI YANG TAMBAH ANGKA
    fmt=".2f",                 # format 2 angka desimal
    annot_kws={"size":9, "color":"black"},
    cbar_kws={"label": "Normalized Performance"}
)

plt.title("Performance Heatmap (Sample vs Variables)", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=9)

plt.tight_layout()
plt.show()
# =========================================================
# 12. PCA BIPLOT 
# =========================================================

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
scores = pca.fit_transform(X_scaled)

loadings = pca.components_.T

features = ['Hmax','CO$_2$max','H$_2$/CO$_2$','Reducing Sugar','pH']

plt.figure(figsize=(8,6))

# =========================================================
# COLOR BY TOPSIS SCORE (IMPORTANT UPGRADE)
# =========================================================
plot_scores = result["TOPSIS_score"].values

scatter = plt.scatter(
    scores[:,0],
    scores[:,1],
    c=plot_scores,
    cmap="viridis",
    s=120,
    edgecolors='black',
    alpha=0.85
)

cbar = plt.colorbar(scatter)
cbar.set_label("TOPSIS Score", fontsize=12)

# =========================================================
# SAMPLE LABELS
# =========================================================
for i, txt in enumerate(samples):
    plt.text(scores[:,0][i], scores[:,1][i], txt, fontsize=9)

# =========================================================
# VARIABLE LOADINGS (SCALED PROPERLY)
# =========================================================
scaling_factor = 2.5

for i in range(len(features)):
    plt.arrow(
        0, 0,
        loadings[i,0]*scaling_factor,
        loadings[i,1]*scaling_factor,
        color='red',
        alpha=0.8,
        head_width=0.05
    )

    plt.text(
        loadings[i,0]*scaling_factor*1.0,
        loadings[i,1]*scaling_factor*1.0,
        features[i],
        fontsize=11,
        color='darkred',
        fontweight='bold'
    )

# =========================================================
# AXIS LABELS
# =========================================================
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=14)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=14)
plt.title("Enhanced PCA Biplot with TOPSIS Integration", fontsize=14)

plt.tight_layout()
plt.show()
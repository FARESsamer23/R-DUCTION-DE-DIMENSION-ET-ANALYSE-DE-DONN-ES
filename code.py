
# -----------------------------------------------------------------------------
# ÉTAPE 0 : IMPORTATION DES BIBLIOTHÈQUES NÉCESSAIRES
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# --- Configuration graphique robuste (compatible toutes versions matplotlib) ---
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')

sns.set_palette("husl")

print("=" * 60)
print("   TP1 — ANALYSE EN COMPOSANTES PRINCIPALES (ACP)")
print("   Dataset : Iris | Source : UCI ML Repository")
print("=" * 60)

# =============================================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES DEPUIS UCI (avec fallback sklearn)
# =============================================================================
print("\n[1] Chargement du dataset Iris depuis UCI...")

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/iris/iris.data"
)
FEATURE_NAMES = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']

try:
    df_raw = pd.read_csv(UCI_URL, header=None,
                         names=FEATURE_NAMES + ['species'])
    # Supprimer les lignes vides éventuelles en fin de fichier
    df_raw.dropna(inplace=True)

    # Encoder la colonne species en entiers (0, 1, 2)
    species_map = {name: i for i, name in
                   enumerate(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])}
    df_raw['species_code'] = df_raw['species'].map(species_map)

    X = df_raw[FEATURE_NAMES]
    y = df_raw['species_code']
    y_name = df_raw['species'].str.replace('Iris-', '')
    print("  ✔ Données chargées depuis UCI ML Repository.")

except Exception as e:
    print(f"  ⚠ Chargement UCI échoué ({e}). Utilisation du dataset sklearn.")
    from sklearn.datasets import load_iris
    iris_sk = load_iris()
    X = pd.DataFrame(iris_sk.data, columns=FEATURE_NAMES)
    y = pd.Series(iris_sk.target)
    y_name = y.map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# --- Aperçu du dataset ---
print(f"\n  Dimensions             : {X.shape[0]} individus × {X.shape[1]} variables")
print(f"  Variables              : {list(X.columns)}")
print(f"  Classes (individus/cl) :\n{y_name.value_counts().to_string()}")
print(f"\n  Aperçu des 5 premières lignes :")
print(pd.concat([X, y_name.rename('species')], axis=1).head().to_string(index=False))

# =============================================================================
# ÉTAPE 2 : STATISTIQUES DESCRIPTIVES
# =============================================================================
print("\n[2] Statistiques descriptives...")
stats = X.describe().T
stats['cv (%)'] = (stats['std'] / stats['mean'] * 100).round(2)
print(stats.round(3).to_string())

# =============================================================================
# ÉTAPE 3 : PRÉTRAITEMENT — STANDARDISATION (Centrage-Réduction)
# =============================================================================
print("\n[3] Standardisation des données (µ=0, σ=1)...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)

print("  Vérification après standardisation :")
print(f"  Moyenne ≈ {X_scaled_df.mean().round(10).values}  (doit être ≈ 0)")
print(f"  Écart-type = {X_scaled_df.std().round(2).values}  (doit être = 1)")

# =============================================================================
# ÉTAPE 4 : MATRICE DE CORRÉLATION (avant ACP)
# =============================================================================
print("\n[4] Matrice de corrélation des variables originales...")

corr_matrix = X.corr()
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.zeros_like(corr_matrix, dtype=bool)
np.fill_diagonal(mask, True)
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap='RdYlGn',
    center=0, square=True, linewidths=0.5,
    cbar_kws={'shrink': 0.8}, ax=ax
)
ax.set_title("Matrice de Corrélation — Variables Iris\n"
             "(Permet de justifier l'utilisation de l'ACP)",
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('fig1_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("  → Forte corrélation entre petal length et petal width : ACP justifiée.")

# =============================================================================
# ÉTAPE 5 : APPLICATION DE L'ACP (toutes composantes)
# =============================================================================
print("\n[5] Application de l'ACP sur les données standardisées...")

pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

variance_ratio   = pca_full.explained_variance_ratio_
cumulative_var   = np.cumsum(variance_ratio)
eigenvalues      = pca_full.explained_variance_
n_components     = len(variance_ratio)

print(f"\n  Valeurs propres (eigenvalues) :")
for i, (ev, vr, cv) in enumerate(zip(eigenvalues, variance_ratio, cumulative_var)):
    marker = " ◄" if cv >= 0.95 and (i == 0 or cumulative_var[i-1] < 0.95) else ""
    print(f"    PC{i+1} : λ={ev:.4f} | Variance={vr:.2%} | Cumulée={cv:.2%}{marker}")

# =============================================================================
# ÉTAPE 6 : CHOIX DU NOMBRE DE DIMENSIONS (Règle de Kaiser + Coude)
# =============================================================================
print("\n[6] Détermination du nombre optimal de composantes...")

# Règle de Kaiser : garder les composantes avec λ > 1
kaiser_components = np.sum(eigenvalues > 1)
# Seuil 95% de variance
threshold_95 = np.argmax(cumulative_var >= 0.95) + 1

print(f"  Règle de Kaiser (λ > 1)     : {kaiser_components} composantes")
print(f"  Seuil 95% de variance       : {threshold_95} composantes")
print(f"  → Choix retenu              : 2 composantes")
print(f"  → Variance conservée        : {cumulative_var[1]:.2%}")

# Scree Plot + Variance Cumulée
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# -- Scree Plot --
bars = ax1.bar(range(1, n_components+1), variance_ratio * 100,
               color=['#2196F3' if i < 2 else '#B0BEC5' for i in range(n_components)],
               alpha=0.85, edgecolor='white', linewidth=1.2)
ax1.plot(range(1, n_components+1), variance_ratio * 100,
         'o--', color='#E53935', linewidth=2, markersize=7, label='Variance (%)')
for bar, vr in zip(bars, variance_ratio):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{vr:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_xlabel('Composantes Principales', fontsize=11)
ax1.set_ylabel('Variance Expliquée (%)', fontsize=11)
ax1.set_title("Éboulis des Valeurs Propres\n(Scree Plot)", fontsize=13, fontweight='bold')
ax1.set_xticks(range(1, n_components+1))
ax1.set_xticklabels([f'PC{i}' for i in range(1, n_components+1)])
ax1.legend(fontsize=10)

# -- Variance Cumulée --
ax2.plot(range(1, n_components+1), cumulative_var * 100,
         's-', color='#43A047', linewidth=2.5, markersize=9, label='Variance cumulée')
ax2.axhline(y=95, color='#E53935', linestyle='--', linewidth=1.5, label='Seuil 95%')
ax2.axhline(y=90, color='#FB8C00', linestyle='--', linewidth=1.5, label='Seuil 90%')
ax2.fill_between(range(1, n_components+1), cumulative_var * 100,
                 alpha=0.15, color='#43A047')
for i, cv in enumerate(cumulative_var):
    ax2.annotate(f'{cv:.1%}',
                 xy=(i+1, cv*100), xytext=(0, 10),
                 textcoords='offset points', ha='center', fontsize=9)
ax2.set_xlabel('Nombre de Composantes', fontsize=11)
ax2.set_ylabel('Variance Cumulée (%)', fontsize=11)
ax2.set_title("Variance Cumulée\n(Critère de choix)", fontsize=13, fontweight='bold')
ax2.set_xticks(range(1, n_components+1))
ax2.set_xticklabels([f'PC{i}' for i in range(1, n_components+1)])
ax2.set_ylim(0, 105)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.4)

plt.suptitle("Analyse de la Variance — Choix du Nombre de Dimensions",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig2_scree_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# ÉTAPE 7 : ACP FINALE À 2 COMPOSANTES
# =============================================================================
print("\n[7] ACP finale avec 2 composantes...")

pca_2d = PCA(n_components=2)
X_2d   = pca_2d.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_2d, columns=['PC1', 'PC2'])
df_pca['species']      = y_name.values
df_pca['species_code'] = y.values

# =============================================================================
# ÉTAPE 8 : VISUALISATION DES INDIVIDUS (Plan Factoriel PC1–PC2)
# =============================================================================
print("\n[8] Projection des individus sur le plan factoriel PC1–PC2...")

COLORS  = {'setosa': '#E53935', 'versicolor': '#43A047', 'virginica': '#1E88E5'}
MARKERS = {'setosa': 'o',       'versicolor': 's',       'virginica': '^'}

fig, ax = plt.subplots(figsize=(10, 8))

for species in TARGET_NAMES:
    mask = df_pca['species'] == species
    ax.scatter(
        df_pca.loc[mask, 'PC1'],
        df_pca.loc[mask, 'PC2'],
        c=COLORS[species],
        marker=MARKERS[species],
        s=90, alpha=0.85, edgecolors='white', linewidth=0.8,
        label=f'Iris {species} (n={mask.sum()})'
    )

# Ellipses de confiance (contours par espèce)
for species in TARGET_NAMES:
    mask = df_pca['species'] == species
    data = df_pca.loc[mask, ['PC1', 'PC2']].values
    mean = data.mean(axis=0)
    cov  = np.cov(data.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle  = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * 2 * np.sqrt(eigvals)
    ell = mpatches.Ellipse(mean, width, height, angle=angle,
                           color=COLORS[species], alpha=0.12, linewidth=1.5,
                           linestyle='--', fill=True, edgecolor=COLORS[species])
    ax.add_patch(ell)

ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
ax.axvline(0, color='grey', linewidth=0.8, linestyle=':')
ax.set_xlabel(f'PC1  ({variance_ratio[0]:.1%} de la variance)', fontsize=12)
ax.set_ylabel(f'PC2  ({variance_ratio[1]:.1%} de la variance)', fontsize=12)
ax.set_title(
    f'Plan Factoriel ACP — Dataset Iris\n'
    f'Variance totale conservée : {cumulative_var[1]:.1%}',
    fontsize=14, fontweight='bold'
)
ax.legend(title="Espèces d'Iris", fontsize=10, title_fontsize=11,
          framealpha=0.9, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_plan_factoriel.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# ÉTAPE 9 : CERCLE DES CORRÉLATIONS
# =============================================================================
print("\n[9] Construction du cercle des corrélations...")

loadings   = pca_2d.components_        # shape (2, 4)
std_pc1    = np.sqrt(pca_2d.explained_variance_[0])
std_pc2    = np.sqrt(pca_2d.explained_variance_[1])

VAR_COLORS = ['#9C27B0', '#FF9800', '#00ACC1', '#F44336']

fig, ax = plt.subplots(figsize=(8, 8))

for i, (var, color) in enumerate(zip(FEATURE_NAMES, VAR_COLORS)):
    x_coord = loadings[0, i] * std_pc1
    y_coord = loadings[1, i] * std_pc2

    ax.annotate(
        "", xy=(x_coord, y_coord), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color=color, lw=2.5)
    )
    offset_x = 0.07 if x_coord >= 0 else -0.07
    offset_y = 0.07 if y_coord >= 0 else -0.07
    ax.text(x_coord + offset_x, y_coord + offset_y,
            var.replace(' (cm)', ''), fontsize=10, color=color,
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec=color))

    # Afficher la corrélation de chaque variable
    corr_pc1 = loadings[0, i]
    corr_pc2 = loadings[1, i]
    print(f"  {var:<25} → PC1: {corr_pc1:+.3f} | PC2: {corr_pc2:+.3f}")

# Cercle unitaire
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), '--', color='steelblue', alpha=0.5, lw=1.5)

ax.axhline(0, color='black', lw=0.8, linestyle='-')
ax.axvline(0, color='black', lw=0.8, linestyle='-')
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.set_xlabel(f'PC1  ({variance_ratio[0]:.1%})', fontsize=12)
ax.set_ylabel(f'PC2  ({variance_ratio[1]:.1%})', fontsize=12)
ax.set_title("Cercle des Corrélations\n"
             "(Interprétation des axes factoriels)",
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_cercle_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# ÉTAPE 10 : BIPLOT (Individus + Variables dans le même espace)
# =============================================================================
print("\n[10] Biplot — Superposition individus & variables...")

fig, ax = plt.subplots(figsize=(11, 9))

# Individus
for species in TARGET_NAMES:
    mask = df_pca['species'] == species
    ax.scatter(
        df_pca.loc[mask, 'PC1'],
        df_pca.loc[mask, 'PC2'],
        c=COLORS[species], marker=MARKERS[species],
        s=70, alpha=0.6, label=f'Iris {species}'
    )

# Variables (flèches mises à l'échelle)
scale = 3.0
for i, (var, color) in enumerate(zip(FEATURE_NAMES, VAR_COLORS)):
    x_coord = loadings[0, i] * scale
    y_coord = loadings[1, i] * scale
    ax.annotate("", xy=(x_coord, y_coord), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax.text(x_coord * 1.12, y_coord * 1.12,
            var.replace(' (cm)', ''), fontsize=9,
            color=color, fontweight='bold', ha='center')

ax.axhline(0, color='grey', lw=0.7, linestyle=':')
ax.axvline(0, color='grey', lw=0.7, linestyle=':')
ax.set_xlabel(f'PC1  ({variance_ratio[0]:.1%})', fontsize=12)
ax.set_ylabel(f'PC2  ({variance_ratio[1]:.1%})', fontsize=12)
ax.set_title("Biplot ACP — Individus & Variables\n"
             "(Analyse simultanée des individus et des variables)",
             fontsize=14, fontweight='bold')
ax.legend(title="Espèces", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_biplot.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# ÉTAPE 11 : CONTRIBUTIONS DES VARIABLES (Loadings)
# =============================================================================
print("\n[11] Contributions des variables aux composantes principales...")

contrib = pd.DataFrame(
    np.abs(loadings.T) ** 2 * 100,
    index=[v.replace(' (cm)', '') for v in FEATURE_NAMES],
    columns=['Contribution PC1 (%)', 'Contribution PC2 (%)']
)
print(contrib.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (ax, pc) in enumerate(zip(axes, ['Contribution PC1 (%)', 'Contribution PC2 (%)'])):
    vals = contrib[pc].sort_values(ascending=True)
    bars = ax.barh(vals.index, vals.values,
                   color=VAR_COLORS[:len(vals)], alpha=0.85, edgecolor='white')
    ax.axvline(100 / len(FEATURE_NAMES), color='red',
               linestyle='--', lw=1.5, label=f'Contribution moyenne ({100/len(FEATURE_NAMES):.1f}%)')
    for bar, val in zip(bars, vals.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)
    ax.set_xlabel('Contribution (%)', fontsize=11)
    ax.set_title(f'Contribution des variables\nà {pc[:2].upper()}{pc[2]}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(vals.values) * 1.2)

plt.suptitle("Contributions des Variables aux Composantes Principales",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig6_contributions.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# ÉTAPE 12 : QUALITÉ DE REPRÉSENTATION — COS² (cos carré)
# =============================================================================
print("\n[12] Qualité de représentation des individus (cos²)...")

cos2_individus = (X_2d ** 2) / np.sum(pca_full.transform(X_scaled) ** 2, axis=1, keepdims=True)
cos2_moyen = cos2_individus.mean(axis=0)
print(f"  cos² moyen sur PC1 : {cos2_moyen[0]:.4f}")
print(f"  cos² moyen sur PC2 : {cos2_moyen[1]:.4f}")
print(f"  cos² total (PC1+PC2) : {cos2_moyen.sum():.4f} → bonne représentation si > 0.7")

# =============================================================================
# ÉTAPE 13 : RECONSTRUCTION ET ERREUR (MSE)
# =============================================================================
print("\n[13] Reconstruction inverse et mesure de la perte d'information...")

X_reconstructed = pca_2d.inverse_transform(X_2d)
mse  = np.mean((X_scaled - X_reconstructed) ** 2)
rmse = np.sqrt(mse)

print(f"  MSE  (Mean Squared Error)  : {mse:.6f}")
print(f"  RMSE (Root MSE)            : {rmse:.6f}")
print(f"  Variance perdue            : {(1 - cumulative_var[1]):.2%}")
print("  → Réduction fidèle : faible erreur confirme la conservation de l'information.")

# =============================================================================
# ÉTAPE 14 : SYNTHÈSE ET INTERPRÉTATION
# =============================================================================
print("\n" + "=" * 60)
print("   SYNTHÈSE — INTERPRÉTATION DES RÉSULTATS ACP")
print("=" * 60)

print(f"""
┌─────────────────────────────────────────────────────────┐
│  RÉSULTATS CLÉS                                         │
├─────────────────────────────────────────────────────────┤
│  PC1 : {variance_ratio[0]:.2%} de la variance                           │
│  PC2 : {variance_ratio[1]:.2%} de la variance                           │
│  PC1 + PC2 : {cumulative_var[1]:.2%} conservés sur 4 dimensions      │
├─────────────────────────────────────────────────────────┤
│  INTERPRÉTATION DES AXES                                │
├─────────────────────────────────────────────────────────┤
│  PC1 : axe de taille globale de la fleur                │
│        → correlé positivement à petal length,           │
│          petal width et sepal length                    │
│        → oppose Iris setosa aux 2 autres espèces        │
│                                                         │
│  PC2 : axe de forme / largeur du sépale                 │
│        → correlé à sepal width                          │
│        → sépare versicolor de virginica                 │
├─────────────────────────────────────────────────────────┤
│  CONCLUSION                                             │
├─────────────────────────────────────────────────────────┤
│  • Setosa : linéairement séparable (cluster compact)    │
│  • Versicolor / Virginica : léger chevauchement en PC2  │
│  • Réduction 4D → 2D validée avec {cumulative_var[1]:.1%} d'info      │
│  • MSE = {mse:.4f} confirme la faible perte               │
└─────────────────────────────────────────────────────────┘
""")

print("\n  Figures sauvegardées :")
figures = [
    "fig1_correlation_matrix.png",
    "fig2_scree_plot.png",
    "fig3_plan_factoriel.png",
    "fig4_cercle_correlations.png",
    "fig5_biplot.png",
    "fig6_contributions.png"
]
for fig_name in figures:
    print(f"    ✔ {fig_name}")

print("\n" + "=" * 60)
print("   ANALYSE ACP TERMINÉE AVEC SUCCÈS")
print("=" * 60)

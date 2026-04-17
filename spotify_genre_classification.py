# =============================================================================
# Spotify 1M Tracks — Genre Classification
# Convex Optimization Project — ML Part
# Models: Logistic Regression | SVM Linear | SVM RBF | MLP Neural Network
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import time

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

# Mettre le chemin vers ton fichier CSV téléchargé depuis Kaggle
df = pd.read_csv("spotify_data.csv")

print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nFirst rows:")
print(df.head(3))
print(f"\nMissing values:\n{df.isnull().sum()}")

# =============================================================================
# 2. PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

# Features audio utilisées pour la classification
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

TARGET = "genre"

# Supprimer les lignes avec valeurs manquantes sur les colonnes utiles
df = df.dropna(subset=AUDIO_FEATURES + [TARGET])

# Nombre de genres
print(f"\nNombre de genres uniques : {df[TARGET].nunique()}")
print(f"Distribution des genres :\n{df[TARGET].value_counts()}")

# Garder les N genres les plus fréquents pour éviter classes trop déséquilibrées
N_GENRES = 10
top_genres = df[TARGET].value_counts().nlargest(N_GENRES).index
df = df[df[TARGET].isin(top_genres)].copy()
print(f"\nAprès filtrage top {N_GENRES} genres : {df.shape[0]} tracks")

# Encoder les labels
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
X = df[AUDIO_FEATURES].values

print(f"\nX shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Classes : {le.classes_}")

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(AUDIO_FEATURES):
    axes[i].boxplot(
        [df[df[TARGET] == g][feature].values for g in le.classes_],
        labels=le.classes_,
        vert=False
    )
    axes[i].set_title(feature)
    axes[i].tick_params(axis='y', labelsize=7)

plt.suptitle("Distribution des features audio par genre", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_features_by_genre.png", dpi=150, bbox_inches="tight")
plt.show()
print("-> Graphique sauvegardé : eda_features_by_genre.png")

# Matrice de corrélation
plt.figure(figsize=(9, 7))
corr = df[AUDIO_FEATURES].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Corrélation entre features audio", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("-> Graphique sauvegardé : eda_correlation.png")

# =============================================================================
# 4. TRAIN / VALIDATION / TEST SPLIT  (70 / 20 / 10)
# =============================================================================
print("\n" + "=" * 60)
print("4. TRAIN / VALIDATION / TEST SPLIT  (70% / 20% / 10%)")
print("=" * 60)

# Étape 1 : séparer 10% pour le test final
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)
# Étape 2 : sur les 90% restants, prendre 20/90 ≈ 22.2% pour la validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2222, random_state=42, stratify=y_temp
)

total = len(y)
print(f"Total     : {total} samples (100%)")
print(f"Train     : {X_train.shape[0]} samples ({X_train.shape[0]/total*100:.1f}%)")
print(f"Validation: {X_val.shape[0]} samples  ({X_val.shape[0]/total*100:.1f}%)")
print(f"Test      : {X_test.shape[0]} samples  ({X_test.shape[0]/total*100:.1f}%)")

# =============================================================================
# 5. MODELS
# =============================================================================
print("\n" + "=" * 60)
print("5. TRAINING MODELS")
print("=" * 60)

results = {}

# ---------------------------------------------------------------------------
# 5.1 LOGISTIC REGRESSION (convexe — log-loss)
# ---------------------------------------------------------------------------
print("\n[1/4] Logistic Regression (convexe)...")
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        solver="lbfgs",        # L-BFGS : quasi-Newton, exploite la convexité
        multi_class="multinomial",
        C=1.0,
        random_state=42
    ))
])
t0 = time.time()
pipe_lr.fit(X_train, y_train)
train_time = time.time() - t0

y_pred_lr_val  = pipe_lr.predict(X_val)
y_pred_lr      = pipe_lr.predict(X_test)
acc_lr_val = accuracy_score(y_val,  y_pred_lr_val)
acc_lr     = accuracy_score(y_test, y_pred_lr)
results["Logistic Regression"] = {"val_accuracy": acc_lr_val, "test_accuracy": acc_lr, "time": train_time}
print(f"   Val Accuracy  : {acc_lr_val:.4f}")
print(f"   Test Accuracy : {acc_lr:.4f} | Temps : {train_time:.2f}s")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# ---------------------------------------------------------------------------
# 5.2 SVM LINÉAIRE (convexe — hinge loss)
# ---------------------------------------------------------------------------
print("\n[2/4] SVM Linéaire (convexe — QP)...")
pipe_svm_lin = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearSVC(
        C=1.0,
        max_iter=2000,
        random_state=42
    ))
])
t0 = time.time()
pipe_svm_lin.fit(X_train, y_train)
train_time = time.time() - t0

y_pred_svm_lin_val = pipe_svm_lin.predict(X_val)
y_pred_svm_lin     = pipe_svm_lin.predict(X_test)
acc_svm_lin_val = accuracy_score(y_val,  y_pred_svm_lin_val)
acc_svm_lin     = accuracy_score(y_test, y_pred_svm_lin)
results["SVM Linéaire"] = {"val_accuracy": acc_svm_lin_val, "test_accuracy": acc_svm_lin, "time": train_time}
print(f"   Val Accuracy  : {acc_svm_lin_val:.4f}")
print(f"   Test Accuracy : {acc_svm_lin:.4f} | Temps : {train_time:.2f}s")
print(classification_report(y_test, y_pred_svm_lin, target_names=le.classes_))

# ---------------------------------------------------------------------------
# 5.3 SVM RBF (convexe dans l'espace dual)
# ---------------------------------------------------------------------------
print("\n[3/4] SVM RBF (convexe — noyau Gaussien)...")

# Sous-échantillonnage pour accélérer (SVM RBF est O(n²))
MAX_TRAIN_SVM = 30_000
if X_train.shape[0] > MAX_TRAIN_SVM:
    idx = np.random.choice(X_train.shape[0], MAX_TRAIN_SVM, replace=False)
    X_train_svm, y_train_svm = X_train[idx], y_train[idx]
    print(f"   (sous-échantillonnage à {MAX_TRAIN_SVM} pour SVM RBF)")
else:
    X_train_svm, y_train_svm = X_train, y_train

pipe_svm_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=42
    ))
])
t0 = time.time()
pipe_svm_rbf.fit(X_train_svm, y_train_svm)
train_time = time.time() - t0

y_pred_svm_rbf_val = pipe_svm_rbf.predict(X_val)
y_pred_svm_rbf     = pipe_svm_rbf.predict(X_test)
acc_svm_rbf_val = accuracy_score(y_val,  y_pred_svm_rbf_val)
acc_svm_rbf     = accuracy_score(y_test, y_pred_svm_rbf)
results["SVM RBF"] = {"val_accuracy": acc_svm_rbf_val, "test_accuracy": acc_svm_rbf, "time": train_time}
print(f"   Val Accuracy  : {acc_svm_rbf_val:.4f}")
print(f"   Test Accuracy : {acc_svm_rbf:.4f} | Temps : {train_time:.2f}s")
print(classification_report(y_test, y_pred_svm_rbf, target_names=le.classes_))

# ---------------------------------------------------------------------------
# 5.4 RÉSEAU DE NEURONES — MLP (non-convexe)
# ---------------------------------------------------------------------------
print("\n[4/4] MLP Neural Network (NON-convexe)...")
pipe_mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=200,
        learning_rate_init=0.001,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    ))
])
t0 = time.time()
pipe_mlp.fit(X_train, y_train)
train_time = time.time() - t0

y_pred_mlp_val = pipe_mlp.predict(X_val)
y_pred_mlp     = pipe_mlp.predict(X_test)
acc_mlp_val = accuracy_score(y_val,  y_pred_mlp_val)
acc_mlp     = accuracy_score(y_test, y_pred_mlp)
results["MLP (Neural Net)"] = {"val_accuracy": acc_mlp_val, "test_accuracy": acc_mlp, "time": train_time}
print(f"   Val Accuracy  : {acc_mlp_val:.4f}")
print(f"   Test Accuracy : {acc_mlp:.4f} | Temps : {train_time:.2f}s")
print(classification_report(y_test, y_pred_mlp, target_names=le.classes_))

# =============================================================================
# 6. COMPARAISON DES MODÈLES
# =============================================================================
print("\n" + "=" * 60)
print("6. COMPARISON OF MODELS")
print("=" * 60)

results_df = pd.DataFrame(results).T
results_df.index.name = "Model"
results_df["val_accuracy"]  = results_df["val_accuracy"].apply(lambda x: f"{x:.4f}")
results_df["test_accuracy"] = results_df["test_accuracy"].apply(lambda x: f"{x:.4f}")
results_df["time"] = results_df["time"].apply(lambda x: f"{x:.2f}s")
print(results_df.to_string())

# Graphique comparaison — Val vs Test
model_names = list(results.keys())
val_accs  = [results[m]["val_accuracy"]  for m in model_names]
test_accs = [results[m]["test_accuracy"] for m in model_names]

x = np.arange(len(model_names))
width = 0.35
colors_val  = ["#64B5F6", "#81C784", "#FFB74D", "#EF9A9A"]
colors_test = ["#1565C0", "#2E7D32", "#E65100", "#B71C1C"]

fig, ax = plt.subplots(figsize=(11, 5))
bars_val  = ax.bar(x - width/2, val_accs,  width, label="Validation (20%)", color=colors_val,  edgecolor="black", linewidth=0.8)
bars_test = ax.bar(x + width/2, test_accs, width, label="Test (10%)",        color=colors_test, edgecolor="black", linewidth=0.8)
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Comparaison des modèles — Val vs Test (70/20/10 split)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
for bar in list(bars_val) + list(bars_test):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("-> Graphique sauvegardé : model_comparison.png")

# =============================================================================
# 7. MATRICE DE CONFUSION — MEILLEUR MODÈLE
# =============================================================================
print("\n" + "=" * 60)
print("7. CONFUSION MATRIX — BEST MODEL")
print("=" * 60)

best_model_name = max(results, key=lambda m: results[m]["accuracy"])
print(f"Meilleur modèle : {best_model_name}")

best_preds = {
    "Logistic Regression": y_pred_lr,
    "SVM Linéaire": y_pred_svm_lin,
    "SVM RBF": y_pred_svm_rbf,
    "MLP (Neural Net)": y_pred_mlp,
}[best_model_name]

cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(11, 9))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
ax.set_title(f"Matrice de confusion — {best_model_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix_best.png", dpi=150, bbox_inches="tight")
plt.show()
print("-> Graphique sauvegardé : confusion_matrix_best.png")

# =============================================================================
# 8. LEARNING CURVE — LOGISTIC REGRESSION (convexité illustrée)
# =============================================================================
print("\n" + "=" * 60)
print("8. LEARNING CURVE — LOGISTIC REGRESSION")
print("=" * 60)

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    pipe_lr, X, y,
    train_sizes=np.linspace(0.05, 1.0, 10),
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#2196F3", label="Train accuracy")
plt.plot(train_sizes, val_scores.mean(axis=1), "s-", color="#F44336", label="Val accuracy")
plt.fill_between(train_sizes,
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color="#2196F3")
plt.fill_between(train_sizes,
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color="#F44336")
plt.xlabel("Taille du training set", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Learning Curve — Logistic Regression (L-BFGS)", fontsize=13, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve_lr.png", dpi=150, bbox_inches="tight")
plt.show()
print("-> Graphique sauvegardé : learning_curve_lr.png")

# =============================================================================
# 9. HYPERPARAMETER TUNING — SVM (C parameter)
# =============================================================================
print("\n" + "=" * 60)
print("9. EFFECT OF C PARAMETER — SVM")
print("=" * 60)

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
svm_accs = []

# Utiliser un sous-set pour la rapidité
MAX_TUNE = 15_000
idx = np.random.choice(X_train.shape[0], min(MAX_TUNE, X_train.shape[0]), replace=False)
X_tune, y_tune = X_train[idx], y_train[idx]

scaler_tune = StandardScaler()
X_tune_sc = scaler_tune.fit_transform(X_tune)
X_test_sc = scaler_tune.transform(X_test)

for C in C_values:
    clf = LinearSVC(C=C, max_iter=2000, random_state=42)
    clf.fit(X_tune_sc, y_tune)
    svm_accs.append(accuracy_score(y_test, clf.predict(X_test_sc)))
    print(f"   C={C:>7} -> accuracy={svm_accs[-1]:.4f}")

plt.figure(figsize=(8, 4))
plt.semilogx(C_values, svm_accs, "o-", color="#4CAF50", linewidth=2, markersize=8)
plt.xlabel("C (paramètre de régularisation)", fontsize=12)
plt.ylabel("Accuracy (test)", fontsize=12)
plt.title("SVM Linéaire — Impact du paramètre C\n(trade-off marge/erreurs)", fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("svm_C_tuning.png", dpi=150, bbox_inches="tight")
plt.show()
print("-> Graphique sauvegardé : svm_C_tuning.png")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 60)
print("RÉSUMÉ FINAL")
print("=" * 60)
print(f"{'Modèle':<22} {'Val Acc':>10} {'Test Acc':>10} {'Convexe':>12}")
print("-" * 58)
convexity = {
    "Logistic Regression": "Oui",
    "SVM Linéaire": "Oui",
    "SVM RBF": "Oui (dual)",
    "MLP (Neural Net)": "NON"
}
for model in model_names:
    va  = results[model]["val_accuracy"]
    ta  = results[model]["test_accuracy"]
    print(f"{model:<22} {va:>10.4f} {ta:>10.4f} {convexity[model]:>12}")
print("=" * 60)
print("Fichiers générés :")
print("  - eda_features_by_genre.png")
print("  - eda_correlation.png")
print("  - model_comparison.png")
print("  - confusion_matrix_best.png")
print("  - learning_curve_lr.png")
print("  - svm_C_tuning.png")

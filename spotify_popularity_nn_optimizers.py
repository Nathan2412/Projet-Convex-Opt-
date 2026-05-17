# %% [markdown]
# # Classification de popularite Spotify par reseau neuronal
#
# Objectif du projet : comparer plusieurs optimiseurs pour l'entrainement
# d'un reseau neuronal sur le dataset Spotify 1 Million Tracks.
#
# Tache ML retenue : classification multiclasse de `popularity` en trois classes :
# - 0 : faible popularite, score 0 a 30
# - 1 : popularite moyenne, score 31 a 70
# - 2 : forte popularite, score 71 a 100

# %%
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, cast

if sys.version_info >= (3, 14):
    raise RuntimeError(
        "TensorFlow n'est pas disponible pour Python 3.14 dans cet environnement. "
        "Creez un environnement virtuel avec Python 3.11 ou 3.12, puis relancez "
        "`python -m pip install -r requirements.txt`."
    )

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
import tensorflow as tf
try:
    from IPython.display import display
except ImportError:
    display = print
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# %%
RANDOM_STATE = 42
DATA_PATH = Path("spotify_data.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mettre a None pour utiliser le dataset complet.
# Garder une valeur plus petite pendant les tests si la machine manque de RAM.
SAMPLE_SIZE: int | None = 250_000

NUMERIC_FEATURES = [
    "year",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
]

CATEGORICAL_FEATURES = ["genre"]

TARGET = "popularity_class"

tf.keras.utils.set_random_seed(RANDOM_STATE)


def make_one_hot_encoder() -> OneHotEncoder:
    """Compatibilite entre scikit-learn recent (`sparse_output`) et ancien (`sparse`)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def to_dense_array(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Convertit la sortie du preprocesseur en tableau dense pour Keras."""
    if sparse.issparse(matrix):
        sparse_matrix = cast(Any, matrix)
        return sparse_matrix.toarray()
    return np.asarray(matrix)


# %% [markdown]
# ## 1. Chargement et exploration rapide

# %%
df = pd.read_csv(DATA_PATH)

# Le CSV Kaggle contient souvent une colonne d'index nommee `Unnamed: 0`.
unnamed_columns = [col for col in df.columns if col.lower().startswith("unnamed")]
if unnamed_columns:
    df = df.drop(columns=unnamed_columns)

print("Dimensions :", df.shape)
print("Colonnes :", df.columns.tolist())
display(df.head())
df.info()
display(df.describe(include="all").T)


# %%
if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Echantillon utilise : {df.shape}")


# %% [markdown]
# ## 2. Visualisations avant modelisation

# %%
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="popularity", bins=30, kde=False)
plt.title("Distribution du score de popularite")
plt.xlabel("Popularity")
plt.ylabel("Nombre de morceaux")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distribution_popularity.png", dpi=150)
plt.show()


# %%
top_genres = df["genre"].value_counts().head(20)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title("Top 20 des genres les plus frequents")
plt.xlabel("Nombre de morceaux")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_genres.png", dpi=150)
plt.show()


# %%
plt.figure(figsize=(11, 8))
corr = df[NUMERIC_FEATURES + ["popularity"]].corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
plt.title("Matrice de correlation des variables numeriques")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150)
plt.show()


# %% [markdown]
# ## 3. Nettoyage et creation de la cible
#
# Colonnes supprimees :
# - `track_id` : identifiant technique sans signification musicale directe ;
# - `track_name` : texte non exploite ici, sauf approche NLP ;
# - `artist_name` : tres forte cardinalite ; on le discute en limite/amelioration ;
# - `popularity` est retire des features apres creation de la classe cible.

# %%
def popularity_class(score: float) -> int:
    if score <= 30:
        return 0
    if score <= 70:
        return 1
    return 2


df = df.drop_duplicates().copy()
df[TARGET] = df["popularity"].apply(popularity_class)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=TARGET)
plt.title("Distribution des classes de popularite")
plt.xlabel("Classe de popularite")
plt.ylabel("Nombre de morceaux")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distribution_popularity_class.png", dpi=150)
plt.show()

print(df[TARGET].value_counts(normalize=True).sort_index())


# %% [markdown]
# ## 4. Split train / validation / test
#
# Le split est stratifie pour conserver les proportions de classes dans chaque ensemble.
# Le preprocesseur sera appris uniquement sur le train set, puis applique aux ensembles
# validation et test.

# %%
feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
X = df[feature_columns].copy()
y = df[TARGET].astype(int).copy()

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)

y_train = y_train.to_numpy(dtype=np.int32)
y_val = y_val.to_numpy(dtype=np.int32)
y_test = y_test.to_numpy(dtype=np.int32)

print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)


# %% [markdown]
# ## 5. Encodage et normalisation

# %%
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)

X_train_prepared = to_dense_array(preprocessor.fit_transform(X_train))
X_val_prepared = to_dense_array(preprocessor.transform(X_val))
X_test_prepared = to_dense_array(preprocessor.transform(X_test))

print("Nombre de features apres preprocessing :", X_train_prepared.shape[1])


# %% [markdown]
# ## 6. Modele neuronal
#
# Architecture de reference :
# - Dense 128, ReLU
# - Dropout
# - Dense 64, ReLU
# - Dropout
# - Dense 32, ReLU
# - Dense 3, softmax
#
# La perte `sparse_categorical_crossentropy` convient car les classes sont codees
# par des entiers 0, 1 et 2.

# %%
def build_model(
    input_dim: int,
    hidden_units: tuple[int, ...] = (128, 64, 32),
    dropout_rate: float = 0.2,
    l2_strength: float = 1e-5,
) -> keras.Model:
    model = keras.Sequential(name="spotify_popularity_classifier")
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_units:
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_strength),
            )
        )
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(3, activation="softmax"))
    return model


def make_optimizer(name: str, learning_rate: float) -> keras.optimizers.Optimizer:
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate)
    if name == "sgd_momentum":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    raise ValueError(f"Optimiseur inconnu : {name}")


# %% [markdown]
# ## 7. Fonction d'entrainement d'une experience

# %%
class_values = np.unique(y_train)
class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=class_values,
    y=y_train,
)
CLASS_WEIGHT = {
    int(class_id): float(weight)
    for class_id, weight in zip(class_values, class_weights_values)
}
print("Class weights :", CLASS_WEIGHT)


# %%
def run_experiment(config: dict) -> dict:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    model = build_model(
        input_dim=X_train_prepared.shape[1],
        hidden_units=config["hidden_units"],
        dropout_rate=config["dropout"],
        l2_strength=config["l2"],
    )

    optimizer = make_optimizer(config["optimizer"], config["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks: list[keras.callbacks.Callback] = [
        cast(
            keras.callbacks.Callback,
            keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            ),
        )
    ]

    start = time.perf_counter()
    history = model.fit(
        X_train_prepared,
        y_train,
        validation_data=(X_val_prepared, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        class_weight=CLASS_WEIGHT,
        verbose=1,
    )
    training_time = time.perf_counter() - start

    val_proba = model.predict(X_val_prepared, batch_size=config["batch_size"])
    val_pred = np.argmax(val_proba, axis=1)

    test_proba = model.predict(X_test_prepared, batch_size=config["batch_size"])
    test_pred = np.argmax(test_proba, axis=1)

    result = {
        "experiment": config["experiment"],
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "architecture": "-".join(map(str, config["hidden_units"])) + "-3",
        "dropout": config["dropout"],
        "epochs_run": len(history.history["loss"]),
        "training_time_sec": training_time,
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_precision_macro": float(precision_score(y_val, val_pred, average="macro", zero_division=0)),
        "val_recall_macro": float(recall_score(y_val, val_pred, average="macro", zero_division=0)),
        "val_f1_macro": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_precision_macro": float(precision_score(y_test, test_pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, test_pred, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        "history": history.history,
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "test_classification_report": classification_report(
            y_test,
            test_pred,
            target_names=["faible", "moyenne", "forte"],
            zero_division=0,
            output_dict=True,
        ),
    }

    model.save(OUTPUT_DIR / f"model_exp_{config['experiment']}_{config['optimizer']}.keras")
    return result


# %% [markdown]
# ## 8. Experiences
#
# Les experiences gardent la meme architecture principale pour comparer les optimiseurs
# de facon equitable. Les dernieres lignes testent aussi l'effet du learning rate
# et du batch size.

# %%
EXPERIMENTS = [
    {
        "experiment": 1,
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "batch_size": 512,
        "hidden_units": (128, 64, 32),
        "dropout": 0.2,
        "l2": 1e-5,
        "epochs": 50,
    },
    {
        "experiment": 2,
        "optimizer": "sgd_momentum",
        "learning_rate": 0.01,
        "batch_size": 512,
        "hidden_units": (128, 64, 32),
        "dropout": 0.2,
        "l2": 1e-5,
        "epochs": 50,
    },
    {
        "experiment": 3,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 512,
        "hidden_units": (128, 64, 32),
        "dropout": 0.2,
        "l2": 1e-5,
        "epochs": 50,
    },
    {
        "experiment": 4,
        "optimizer": "rmsprop",
        "learning_rate": 0.001,
        "batch_size": 512,
        "hidden_units": (128, 64, 32),
        "dropout": 0.2,
        "l2": 1e-5,
        "epochs": 50,
    },
    {
        "experiment": 5,
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "batch_size": 512,
        "hidden_units": (128, 64, 32),
        "dropout": 0.2,
        "l2": 1e-5,
        "epochs": 50,
    },
    {
        "experiment": 6,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "batch_size": 256,
        "hidden_units": (128, 64, 32),
        "dropout": 0.2,
        "l2": 1e-5,
        "epochs": 50,
    },
]

results = []
for config in EXPERIMENTS:
    print(f"\n=== Experience {config['experiment']} : {config['optimizer']} ===")
    results.append(run_experiment(config))


# %% [markdown]
# ## 9. Tableau comparatif des resultats

# %%
summary_columns = [
    "experiment",
    "optimizer",
    "learning_rate",
    "batch_size",
    "architecture",
    "dropout",
    "epochs_run",
    "training_time_sec",
    "best_val_loss",
    "best_val_accuracy",
    "val_f1_macro",
    "test_accuracy",
    "test_f1_macro",
]

summary_df = pd.DataFrame([{k: r[k] for k in summary_columns} for r in results])
summary_df = summary_df.sort_values(["val_f1_macro", "best_val_loss"], ascending=[False, True])
display(summary_df)
summary_df.to_csv(OUTPUT_DIR / "optimizer_comparison_results.csv", index=False)

class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

with open(OUTPUT_DIR / "full_results.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, cls=_NumpyEncoder)


# %% [markdown]
# ## 10. Courbes d'apprentissage

# %%
def plot_histories(metric: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(10, 6))
    for result in results:
        label = (
            f"Exp {result['experiment']} - {result['optimizer']} "
            f"(lr={result['learning_rate']}, batch={result['batch_size']})"
        )
        plt.plot(result["history"][metric], linestyle="-", label=f"train {label}")
        plt.plot(result["history"][f"val_{metric}"], linestyle="--", label=f"val {label}")
    plt.title(ylabel)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.show()


plot_histories("loss", "Loss train / validation", "optimizer_loss_curves.png")
plot_histories("accuracy", "Accuracy train / validation", "optimizer_accuracy_curves.png")


# %%
summary_df["experiment"] = summary_df["experiment"].astype(str)
plt.figure(figsize=(9, 5))
sns.barplot(data=summary_df, x="optimizer", y="val_f1_macro", hue="experiment")
plt.title("F1-score macro de validation par optimiseur")
plt.xlabel("Optimiseur")
plt.ylabel("F1-score macro validation")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "val_f1_by_optimizer.png", dpi=150)
plt.show()


# %% [markdown]
# ## 11. Matrice de confusion du meilleur modele

# %%
best_result = summary_df.iloc[0]
best_experiment_id = int(best_result["experiment"])
best_full_result = next(r for r in results if r["experiment"] == best_experiment_id)

cm = np.array(best_full_result["test_confusion_matrix"])
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["faible", "moyenne", "forte"],
    yticklabels=["faible", "moyenne", "forte"],
)
plt.title(f"Matrice de confusion - meilleure experience {best_experiment_id}")
plt.xlabel("Classe predite")
plt.ylabel("Classe reelle")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "best_confusion_matrix.png", dpi=150)
plt.show()

pd.DataFrame(best_full_result["test_classification_report"]).T


# %% [markdown]
# ## 12. Interpretation a completer
#
# Points a commenter dans le rapport :
# - l'optimiseur qui atteint la plus faible validation loss ;
# - l'optimiseur qui obtient le meilleur F1-score macro ;
# - la vitesse de convergence observee sur les courbes ;
# - les ecarts entre train et validation pour detecter le surapprentissage ;
# - les classes les plus confondues dans la matrice de confusion ;
# - la coherence entre les resultats experimentaux et la theorie des optimiseurs ;
# - les limites liees au fait que la popularite depend aussi de facteurs externes.

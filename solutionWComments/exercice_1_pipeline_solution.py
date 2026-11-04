"""
SOLUTION COMMENTÉE - EXERCICE 1 : Pipeline de prétraitement + modélisation
==========================================================================
Objectif : prédire le churn (résiliation) d'un client à partir de variables
numériques, catégorielles nominales et ordinales, avec valeurs manquantes.

Idée centrale : TOUT le prétraitement est placé DANS un Pipeline, afin que
chaque transformation soit "apprise" uniquement sur les données d'entraînement
(pas de fuite de données) et que le tout soit reproductible.
"""

# --- Imports ---------------------------------------------------------------
import numpy as np                                   # calcul numérique, générateur aléatoire
import pandas as pd                                  # manipulation de tableaux (DataFrame)
from sklearn.model_selection import train_test_split, cross_val_score
# train_test_split : sépare les données en train/test
# cross_val_score  : évalue un modèle par validation croisée
from sklearn.compose import ColumnTransformer        # applique des traitements différents selon les colonnes
from sklearn.pipeline import Pipeline                # enchaîne des étapes (transformations + modèle)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
# StandardScaler : centre-réduit les variables numériques
# OneHotEncoder  : encode les catégories SANS ordre (une colonne 0/1 par modalité)
# OrdinalEncoder : encode les catégories AVEC ordre (0, 1, 2...)
from sklearn.impute import SimpleImputer             # remplace les valeurs manquantes (NaN)
from sklearn.ensemble import RandomForestClassifier  # modèle d'ensemble d'arbres de décision


# --- Génération du jeu de données (fourni par l'énoncé) --------------------
def generer_donnees(n=1000, seed=42):
    rng = np.random.default_rng(seed)                # générateur aléatoire avec graine -> résultats reproductibles
    df = pd.DataFrame({
        "age": rng.integers(18, 75, n).astype(float),            # float pour pouvoir y mettre des NaN
        "revenu": rng.normal(2500, 900, n),                      # revenu ~ loi normale
        "anciennete_mois": rng.integers(1, 120, n).astype(float),
        "nb_reclamations": rng.poisson(1.5, n).astype(float),    # comptage -> loi de Poisson
        "type_contrat": rng.choice(["mensuel", "annuel", "biannuel"], n),   # catégorielle nominale
        "ville": rng.choice(["Tunis", "Sfax", "Sousse", "Bizerte"], n),     # catégorielle nominale
        "niveau_service": rng.choice(["basique", "standard", "premium"], n),  # catégorielle ordinale
    })
    # Score latent : combinaison de variables qui "explique" le churn (la vraie règle cachée)
    score = (
        -0.02 * df["anciennete_mois"]                         # plus d'ancienneté -> moins de churn
        + 0.5 * df["nb_reclamations"]                         # plus de réclamations -> plus de churn
        + (df["type_contrat"] == "mensuel") * 1.0             # contrat mensuel -> plus de churn
        - (df["niveau_service"] == "premium") * 0.7           # service premium -> moins de churn
        + rng.normal(0, 0.5, n)                               # bruit aléatoire pour rendre le problème réaliste
    )
    df["churn"] = (score > score.median()).astype(int)        # cible binaire : 1 si score au-dessus de la médiane

    # On dégrade volontairement les données pour s'entraîner au nettoyage
    for col in ["age", "revenu", "type_contrat"]:
        masque = rng.random(n) < 0.08                         # ~8 % de lignes tirées au hasard
        df.loc[masque, col] = np.nan                          # -> valeurs manquantes
    df.loc[5, "revenu"] = 250000                              # une valeur aberrante
    return df


df = generer_donnees()

# --- Q1. Exploration rapide ------------------------------------------------
# Toujours regarder les données AVANT de modéliser : taille, trous, équilibre des classes.
print("Dimensions :", df.shape)
print("\nValeurs manquantes :\n", df.isna().sum())          # détermine quelles colonnes nécessitent une imputation
print("\nRépartition de la cible :\n", df["churn"].value_counts(normalize=True))
# La répartition des classes indique si le problème est déséquilibré
# (ici 50/50, donc l'accuracy est une métrique acceptable).

# --- Q2. Séparation X / y et train / test ----------------------------------
X = df.drop(columns="churn")        # variables explicatives (toutes sauf la cible)
y = df["churn"]                     # variable cible à prédire
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,                  # 20 % des données gardées pour le test final (hold-out)
    stratify=y,                     # conserve la même proportion de classes dans train et test
    random_state=42,                # rend la séparation reproductible
)
# Le jeu de test n'est JAMAIS utilisé pendant l'entraînement : il simule des données futures inconnues.

# --- Q3. Identification des types de colonnes ------------------------------
col_ordinale = ["niveau_service"]                 # a un ordre naturel : basique < standard < premium
col_nominales = ["type_contrat", "ville"]         # pas d'ordre -> One-Hot pour ne pas inventer un ordre artificiel
col_numeriques = X.select_dtypes(include="number").columns.tolist()  # détection automatique des colonnes numériques
print("\nNumériques :", col_numeriques)

# --- Q4. Transformateurs + ColumnTransformer -------------------------------
# a) Colonnes numériques : imputation puis mise à l'échelle
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # médiane = robuste à l'aberrant (250000), contrairement à la moyenne
    ("scaler", StandardScaler()),                    # ramène chaque variable à moyenne 0 / écart-type 1 (ordres de grandeur comparables)
])
# b) Colonnes nominales : imputation puis One-Hot
nom_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),     # pour du texte on ne peut pas faire de médiane -> modalité la plus fréquente
    ("onehot", OneHotEncoder(handle_unknown="ignore")),       # "ignore" évite une erreur si une modalité inconnue apparaît en production
])
# c) Colonne ordinale : encodage avec ordre explicite
ord_transformer = Pipeline([
    ("ordinal", OrdinalEncoder(categories=[["basique", "standard", "premium"]])),
    # on impose l'ordre : sans cela, sklearn classerait par ordre alphabétique (basique, premium, standard) -> faux
])

# ColumnTransformer : applique chaque transformateur à ses colonnes et concatène les résultats
preprocessor = ColumnTransformer([
    ("num", num_transformer, col_numeriques),
    ("nom", nom_transformer, col_nominales),
    ("ord", ord_transformer, col_ordinale),
])

# --- Q5. Pipeline complet --------------------------------------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),                  # étape 1 : préparer les données
    ("classifier", RandomForestClassifier(
        n_estimators=200,                            # 200 arbres : plus stable qu'un seul arbre
        random_state=42,                             # reproductibilité
    )),                                              # étape 2 : modèle
])
# Avantage : un seul objet à entraîner, évaluer, sauvegarder et déployer.

# --- Q6. Validation croisée (sur le train uniquement) ----------------------
# Le train est découpé en 5 plis ; le modèle est entraîné 5 fois, chaque fois évalué sur le pli restant.
# Comme le pipeline est passé en entier, imputer/scaler sont ré-ajustés à chaque pli
# uniquement sur les données d'entraînement du pli -> aucune fuite.
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"\nCV accuracy : {scores.mean():.4f} (+/- {scores.std():.4f})")
# La moyenne estime la performance ; l'écart-type indique la stabilité du modèle.

# --- Q7. Entraînement final et test ----------------------------------------
pipeline.fit(X_train, y_train)                       # entraînement sur tout le train
print(f"Accuracy sur le test : {pipeline.score(X_test, y_test):.4f}")
# Le test n'est utilisé qu'UNE fois, à la fin : c'est l'estimation honnête de la performance réelle.

# --- Q8. Réponse à la question de réflexion --------------------------------
# Si on impute/scale sur tout X avant le split, la médiane et la moyenne/écart-type
# sont calculées avec des données du jeu de test : de l'information du test "fuit"
# dans l'entraînement (DATA LEAKAGE) et le score obtenu est optimiste.
# Dans un Pipeline, chaque transformateur est ajusté (fit) uniquement sur les données
# d'entraînement de chaque pli, puis appliqué (transform) à la validation/test.
# Cela garantit aussi la reproductibilité : un seul objet à sauvegarder et à déployer.

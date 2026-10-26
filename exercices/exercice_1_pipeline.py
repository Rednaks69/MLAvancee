"""
EXERCICE 1 : Construire un pipeline de prétraitement + modélisation
===================================================================
Notions couvertes (6.1) : imputation, encodage catégoriel, scaling,
ColumnTransformer, Pipeline, hold-out, validation croisée, data leakage.

Contexte : un jeu de données "clients" contient des variables numériques,
des variables catégorielles et des valeurs manquantes. On veut prédire
si un client va résilier son abonnement (churn = 1) ou non (churn = 0).

Le jeu de données est généré pour vous (ne pas modifier cette partie).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------
# Génération du jeu de données (NE PAS MODIFIER)
# ---------------------------------------------------------------
def generer_donnees(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 75, n).astype(float),
        "revenu": rng.normal(2500, 900, n),
        "anciennete_mois": rng.integers(1, 120, n).astype(float),
        "nb_reclamations": rng.poisson(1.5, n).astype(float),
        "type_contrat": rng.choice(["mensuel", "annuel", "biannuel"], n),
        "ville": rng.choice(["Tunis", "Sfax", "Sousse", "Bizerte"], n),
        "niveau_service": rng.choice(["basique", "standard", "premium"], n),
    })
    # Score latent -> variable cible
    score = (
        -0.02 * df["anciennete_mois"]
        + 0.5 * df["nb_reclamations"]
        + (df["type_contrat"] == "mensuel") * 1.0
        - (df["niveau_service"] == "premium") * 0.7
        + rng.normal(0, 0.5, n)
    )
    df["churn"] = (score > score.median()).astype(int)

    # Injection de valeurs manquantes et d'un aberrant
    for col in ["age", "revenu", "type_contrat"]:
        masque = rng.random(n) < 0.08
        df.loc[masque, col] = np.nan
    df.loc[5, "revenu"] = 250000  # valeur aberrante
    return df


df = generer_donnees()

# ---------------------------------------------------------------
# QUESTIONS
# ---------------------------------------------------------------

# Q1. Affichez les dimensions, le nombre de valeurs manquantes par colonne
#     et la répartition de la variable cible.
# TODO


# Q2. Séparez X (toutes les colonnes sauf 'churn') et y ('churn').
#     Faites un split train/test 80/20 STRATIFIÉ avec random_state=42.
# TODO


# Q3. Identifiez automatiquement :
#       - les colonnes numériques
#       - les colonnes catégorielles NOMINALES (type_contrat, ville)
#       - la colonne catégorielle ORDINALE (niveau_service : basique < standard < premium)
# TODO


# Q4. Construisez trois transformateurs (Pipeline) :
#       - numérique   : imputation par la médiane + StandardScaler
#       - nominal     : imputation par la valeur la plus fréquente + OneHotEncoder(handle_unknown="ignore")
#       - ordinal     : OrdinalEncoder avec l'ordre explicite des catégories
#     Puis regroupez-les dans un ColumnTransformer.
# TODO


# Q5. Construisez le pipeline complet (préprocesseur + RandomForestClassifier
#     avec n_estimators=200, random_state=42).
# TODO


# Q6. Évaluez le pipeline par validation croisée (5 plis, scoring="accuracy")
#     UNIQUEMENT sur le jeu d'entraînement. Affichez la moyenne et l'écart-type.
# TODO


# Q7. Entraînez le pipeline sur tout le train et affichez l'accuracy sur le test.
# TODO


# Q8 (réflexion, répondez en commentaire) :
#     Pourquoi est-il plus correct de mettre le scaler et l'imputer DANS le pipeline
#     plutôt que de les appliquer sur tout X avant le split ? Quel risque évite-t-on ?

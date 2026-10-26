"""
EXERCICE 3 : Étude de cas de bout en bout (End-to-End) - Maintenance prédictive
===============================================================================
Notions couvertes (6.1 + 6.2 + 6.3) : split stratifié, pipeline, comparaison
de modèles, GridSearchCV, AUC-ROC, évaluation finale, interprétation,
sauvegarde du modèle et mini-application de prédiction.

Mission : construire un outil d'aide à la décision industrielle qui prédit
la panne d'une machine (classe 1 = panne, classe 0 = fonctionnement normal)
à partir de mesures de capteurs. Les pannes sont rares (~10 %), d'où
l'utilisation de class_weight="balanced".
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve,
                             classification_report, confusion_matrix)

NOMS_CLASSES = ["fonctionnement", "panne"]


# ---------------------------------------------------------------
# Génération du jeu de données (NE PAS MODIFIER)
# ---------------------------------------------------------------
def generer_donnees(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "temperature_air": rng.normal(300, 2, n),
        "temperature_process": rng.normal(310, 1.5, n),
        "vitesse_rotation": rng.normal(1500, 150, n),
        "couple": rng.normal(40, 10, n),
        "usure_outil_min": rng.integers(0, 250, n),
        "vibration": rng.gamma(2.0, 1.0, n),
    })
    score = (0.02 * df["usure_outil_min"]
             + 0.06 * df["couple"]
             + 0.5 * df["vibration"]
             + 0.4 * (df["temperature_process"] - df["temperature_air"] - 10)
             + rng.normal(0, 0.8, n))
    df["panne"] = (score > score.quantile(0.90)).astype(int)
    return df


# ---------------------------------------------------------------
# ÉTAPE 1 : Acquisition et partitionnement
# ---------------------------------------------------------------
def charger_et_separer():
    """Génère les données et retourne
    X_train, X_test, y_train, y_test, noms_des_variables, NOMS_CLASSES.
    Split 75/25, stratifié, random_state=42."""
    # TODO
    pass


# ---------------------------------------------------------------
# ÉTAPE 2 : Comparaison de deux modèles par GridSearchCV
# ---------------------------------------------------------------
def optimiser_modeles(X_train, y_train):
    """Construit DEUX pipelines (scaler + modèle, avec class_weight='balanced')
    et les optimise chacun avec GridSearchCV (cv=5, scoring='roc_auc', n_jobs=-1) :
        - Régression logistique : C dans [0.01, 0.1, 1, 10]
        - Random Forest : n_estimators [100, 200], max_depth [None, 5, 10]
    Retourne un dictionnaire {nom: GridSearchCV déjà entraîné}."""
    # TODO
    pass


# ---------------------------------------------------------------
# ÉTAPE 3 : Évaluation finale sur le jeu de test
# ---------------------------------------------------------------
def evaluer(resultats, X_test, y_test):
    """Pour chaque modèle : affiche meilleurs paramètres, AUC-ROC CV, AUC-ROC test.
    Sauvegarde les courbes ROC des deux modèles dans 'roc.png'.
    Retourne le nom du meilleur modèle (AUC test la plus élevée)."""
    # TODO
    pass


# ---------------------------------------------------------------
# ÉTAPE 4 : Interprétation du meilleur modèle
# ---------------------------------------------------------------
def interpreter(meilleur_pipeline, X_test, y_test, noms_variables, noms_classes):
    """Affiche la matrice de confusion, le rapport de classification et les 3
    variables les plus influentes :
        - si le modèle final est un RandomForest : feature_importances_
        - si c'est une régression logistique : valeurs absolues des coefficients"""
    # TODO
    pass


# ---------------------------------------------------------------
# ÉTAPE 5 : Mini-application
# ---------------------------------------------------------------
def sauvegarder_modele(modele, chemin="modele_maintenance.joblib"):
    """Sauvegarde le pipeline complet avec joblib."""
    # TODO
    pass


def predire_machine(chemin, mesures_df, seuil_panne=0.3):
    """Charge le modèle sauvegardé et retourne un dict :
        {"proba_panne": float,
         "decision": "ALERTE : maintenance préventive" ou "Fonctionnement normal"}
    La décision est 'ALERTE' si proba_panne >= seuil_panne (seuil volontairement bas
    pour favoriser le rappel). mesures_df est un DataFrame d'une seule ligne."""
    # TODO
    pass


def run_case_study():
    X_train, X_test, y_train, y_test, noms_vars, noms_classes = charger_et_separer()
    resultats = optimiser_modeles(X_train, y_train)
    meilleur_nom = evaluer(resultats, X_test, y_test)
    meilleur = resultats[meilleur_nom].best_estimator_
    interpreter(meilleur, X_test, y_test, noms_vars, noms_classes)
    sauvegarder_modele(meilleur)

    # Simulation : on prend la 1ère machine du jeu de test
    machine = X_test.iloc[[0]]
    print("\nPrédiction mini-app :", predire_machine("modele_maintenance.joblib", machine))
    print("Vraie classe :", noms_classes[y_test.iloc[0]])


if __name__ == "__main__":
    run_case_study()

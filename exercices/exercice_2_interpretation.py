"""
EXERCICE 2 : Évaluer et interpréter un modèle (risque de crédit)
================================================================
Notions couvertes (6.2) : matrice de confusion, précision, rappel, F1,
rapport de classification, importance des variables (Gini),
choix du seuil de décision en contexte sensible (finance).

Contexte : une banque veut prédire si un client va faire DÉFAUT de paiement
(defaut = 1) ou rembourser normalement (defaut = 0).
Classe positive = défaut. Environ 20 % des clients font défaut.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sauvegarde les figures sans ouvrir de fenêtre
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score,
                             recall_score, f1_score)


# ---------------------------------------------------------------
# Génération du jeu de données (NE PAS MODIFIER)
# ---------------------------------------------------------------
def generer_donnees(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    revenu = rng.normal(2800, 1000, n).clip(800, None)
    montant = rng.normal(15000, 7000, n).clip(1000, None)
    duree = rng.choice([12, 24, 36, 48, 60], n)
    df = pd.DataFrame({
        "age": rng.integers(21, 70, n),
        "revenu_mensuel": revenu,
        "montant_pret": montant,
        "duree_mois": duree,
        "ratio_mensualite_revenu": (montant / duree) / revenu,
        "nb_retards_paiement": rng.poisson(1.0, n),
        "anciennete_emploi_ans": rng.integers(0, 35, n),
        "nb_credits_actifs": rng.integers(0, 6, n),
    })
    score = (6 * df["ratio_mensualite_revenu"]
             + 0.5 * df["nb_retards_paiement"]
             + 0.15 * df["nb_credits_actifs"]
             - 0.05 * df["anciennete_emploi_ans"]
             + rng.normal(0, 0.7, n))
    df["defaut"] = (score > score.quantile(0.80)).astype(int)
    return df


df = generer_donnees()
noms_variables = [c for c in df.columns if c != "defaut"]
X = df[noms_variables]
y = df["defaut"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
])
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------
# QUESTIONS
# ---------------------------------------------------------------

# Q1. Prédisez les classes du jeu de test et affichez la matrice de confusion.
#     Extrayez TN, FP, FN, TP avec .ravel().
# TODO


# Q2. Calculez À LA MAIN (avec TP, FP, FN) la précision, le rappel et le F1
#     pour la classe positive (défaut). Vérifiez avec precision_score,
#     recall_score, f1_score.
# TODO


# Q3. Affichez le classification_report avec les noms de classes
#     ["remboursé", "défaut"] et l'accuracy globale.
#     Pourquoi l'accuracy seule serait-elle trompeuse ici ?
# TODO


# Q4. Récupérez les importances du RandomForest (étape 'rf' du pipeline).
#     Affichez le top 5 des variables et sauvegardez un graphique en barres
#     horizontales de toutes les variables dans 'importances.png'.
# TODO


# Q5. Contexte financier : accorder un prêt à un client qui fera défaut (faux
#     négatif) coûte très cher à la banque.
#       - obtenez la probabilité de défaut avec predict_proba
#       - comparez les seuils 0.5, 0.3 et 0.2 : pour chacun, affichez
#         rappel, précision et nombre de faux négatifs
# TODO


# Q6 (réflexion, répondez en commentaire) :
#     a) Quelle métrique privilégier ici : précision ou rappel de la classe "défaut" ?
#        Quel est le coût d'un abaissement du seuil pour la banque ?
#     b) L'importance de Gini a-t-elle des limites ? Pensez aux variables
#        'montant_pret', 'duree_mois', 'revenu_mensuel' et 'ratio_mensualite_revenu'.

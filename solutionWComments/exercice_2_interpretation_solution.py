"""
SOLUTION COMMENTÉE - EXERCICE 2 : Évaluer et interpréter un modèle (risque de crédit)
=====================================================================================
Objectif : aller au-delà de "le modèle a X % de bonnes réponses" et comprendre
 - quelles erreurs il commet (matrice de confusion, précision, rappel, F1),
 - sur quelles variables il s'appuie (importance de Gini),
 - comment régler le seuil de décision selon le coût des erreurs (finance).
"""

# --- Imports ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # backend sans fenêtre : les figures sont sauvegardées en fichier
import matplotlib.pyplot as plt        # tracé de graphiques
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score,
                             recall_score, f1_score)   # toutes les métriques d'évaluation utilisées plus bas


# --- Génération du jeu de données (fourni par l'énoncé) --------------------
def generer_donnees(n=2000, seed=42):
    rng = np.random.default_rng(seed)                       # graine fixe -> mêmes données à chaque exécution
    revenu = rng.normal(2800, 1000, n).clip(800, None)      # clip : pas de revenu inférieur à 800
    montant = rng.normal(15000, 7000, n).clip(1000, None)   # montant du prêt, minimum 1000
    duree = rng.choice([12, 24, 36, 48, 60], n)             # durée du prêt en mois
    df = pd.DataFrame({
        "age": rng.integers(21, 70, n),
        "revenu_mensuel": revenu,
        "montant_pret": montant,
        "duree_mois": duree,
        "ratio_mensualite_revenu": (montant / duree) / revenu,   # variable DÉRIVÉE : mensualité / revenu (corrélée aux 3 précédentes)
        "nb_retards_paiement": rng.poisson(1.0, n),
        "anciennete_emploi_ans": rng.integers(0, 35, n),
        "nb_credits_actifs": rng.integers(0, 6, n),
    })
    # Score de risque caché : plus il est élevé, plus le défaut est probable
    score = (6 * df["ratio_mensualite_revenu"]
             + 0.5 * df["nb_retards_paiement"]
             + 0.15 * df["nb_credits_actifs"]
             - 0.05 * df["anciennete_emploi_ans"]
             + rng.normal(0, 0.7, n))                       # bruit
    df["defaut"] = (score > score.quantile(0.80)).astype(int)   # les 20 % de scores les plus hauts font défaut
    return df


# --- Préparation (fournie par l'énoncé) ------------------------------------
df = generer_donnees()
noms_variables = [c for c in df.columns if c != "defaut"]   # liste des variables explicatives
X = df[noms_variables]
y = df["defaut"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42        # stratify : garde ~20 % de défauts dans train ET test
)
pipeline = Pipeline([
    ("scaler", StandardScaler()),                           # mise à l'échelle (utile surtout pour d'autres modèles ; inoffensive pour un RF)
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
])
pipeline.fit(X_train, y_train)                              # entraînement sur le train uniquement

# --- Q1. Matrice de confusion ----------------------------------------------
y_pred = pipeline.predict(X_test)                # classes prédites (0 ou 1) sur des données jamais vues
cm = confusion_matrix(y_test, y_pred)            # tableau 2x2 : lignes = réalité, colonnes = prédiction
tn, fp, fn, tp = cm.ravel()                      # aplatit la matrice dans l'ordre TN, FP, FN, TP
print("Matrice de confusion :\n", cm)
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
# TN : bons payeurs bien identifiés | FP : bons payeurs refusés à tort
# FN : défauts non détectés (le plus coûteux) | TP : défauts bien détectés

# --- Q2. Précision, rappel, F1 calculés à la main --------------------------
precision = tp / (tp + fp)       # parmi les clients prédits "défaut", quelle part l'est vraiment ? (fiabilité des alertes)
rappel = tp / (tp + fn)          # parmi les vrais défauts, quelle part a été détectée ? (capacité de détection)
f1 = 2 * precision * rappel / (precision + rappel)   # moyenne harmonique : forte seulement si précision ET rappel sont élevés
print(f"\nMain      : précision={precision:.4f}  rappel={rappel:.4f}  F1={f1:.4f}")
print(f"Sklearn   : précision={precision_score(y_test, y_pred):.4f}  "
      f"rappel={recall_score(y_test, y_pred):.4f}  F1={f1_score(y_test, y_pred):.4f}")
# La comparaison avec sklearn vérifie qu'on a bien compris les formules (résultats identiques).

# --- Q3. Rapport de classification -----------------------------------------
print("\n--- Rapport de classification ---")
print(classification_report(y_test, y_pred, target_names=["remboursé", "défaut"]))
# Donne précision/rappel/F1 pour CHAQUE classe d'un seul coup, avec les noms lisibles.
print(f"Accuracy globale : {accuracy_score(y_test, y_pred):.4f}")
# L'accuracy est trompeuse car les classes sont déséquilibrées (80 % / 20 %) :
# un modèle qui prédit toujours "remboursé" aurait déjà ~80 % d'accuracy
# sans détecter aucun défaut.

# --- Q4. Importance des variables ------------------------------------------
rf = pipeline.named_steps["rf"]                  # on récupère le modèle à l'intérieur du pipeline par son nom d'étape
importances = rf.feature_importances_            # contribution de chaque variable à la réduction de l'impureté de Gini
indices = np.argsort(importances)[::-1]          # indices triés du plus important au moins important

print("\n--- Top 5 des variables ---")
for rang in range(5):
    idx = indices[rang]
    print(f"{rang + 1}. {noms_variables[idx]} (importance : {importances[idx]:.4f})")

ordre = indices[::-1]                            # ordre inversé : barh dessine de bas en haut, donc la plus importante finit en haut
plt.figure(figsize=(8, 5))
plt.barh([noms_variables[i] for i in ordre], importances[ordre])   # barres horizontales : noms longs plus lisibles
plt.xlabel("Importance (Gini)")
plt.title("Importance des variables - Random Forest (risque de crédit)")
plt.tight_layout()                               # évite que les libellés soient coupés
plt.savefig("importances.png", dpi=120)          # sauvegarde en fichier image
plt.close()                                      # libère la mémoire de la figure

# --- Q5. Choix du seuil de décision ----------------------------------------
# predict() utilise implicitement un seuil de 0.5. Or ici les deux erreurs n'ont pas le même coût :
# on récupère donc les probabilités pour choisir nous-mêmes le seuil.
proba_defaut = pipeline.predict_proba(X_test)[:, 1]      # colonne 1 = probabilité d'appartenir à la classe "défaut"

print("\n--- Effet du seuil (classe positive = défaut) ---")
for seuil in [0.5, 0.3, 0.2]:
    pred = (proba_defaut >= seuil).astype(int)           # "défaut" si la probabilité dépasse le seuil
    r = recall_score(y_test, pred)
    p = precision_score(y_test, pred)
    faux_neg = int(((y_test == 1) & (pred == 0)).sum())  # défauts réels que le modèle a laissés passer
    print(f"seuil={seuil:.1f} -> rappel={r:.3f}  précision={p:.3f}  faux négatifs={faux_neg}")
# Observation : plus le seuil baisse, plus le rappel monte (moins de défauts manqués)
# mais plus la précision baisse (plus de bons clients refusés).

# --- Q6. Réponses ----------------------------------------------------------
# a) Le RAPPEL de la classe "défaut" : manquer un client qui ne remboursera pas
#    (faux négatif) coûte le capital prêté, alors qu'un faux positif ne coûte
#    qu'un client refusé à tort (manque à gagner sur les intérêts). Abaisser le
#    seuil détecte plus de défauts mais refuse davantage de bons clients : la
#    précision baisse. Le seuil se choisit selon le coût relatif des deux erreurs.
# b) Oui : 'montant_pret', 'duree_mois' et 'revenu_mensuel' sont corrélées au
#    'ratio_mensualite_revenu' (qui est calculé à partir d'elles). L'importance
#    de Gini se répartit entre ces variables redondantes et peut sous-estimer
#    chacune d'elles ; elle est aussi biaisée en faveur des variables continues
#    et calculée sur le train. Alternatives : permutation importance sur le test,
#    valeurs SHAP.

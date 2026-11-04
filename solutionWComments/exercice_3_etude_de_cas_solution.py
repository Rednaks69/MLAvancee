"""
SOLUTION COMMENTÉE - EXERCICE 3 : Étude de cas de bout en bout - Maintenance prédictive
=======================================================================================
Objectif : construire un petit outil complet, de la donnée brute à la décision :
  1. acquérir et séparer les données,
  2. comparer deux modèles en optimisant leurs hyperparamètres,
  3. évaluer honnêtement sur un jeu de test,
  4. interpréter le meilleur modèle,
  5. le sauvegarder et l'utiliser comme une mini-application.

Chaque étape est une fonction : le code est lisible, testable et réutilisable.
"""

# --- Imports ---------------------------------------------------------------
import numpy as np
import pandas as pd
import joblib                                  # sauvegarde / chargement de modèles sklearn sur disque
import matplotlib
matplotlib.use("Agg")                          # figures sauvegardées en fichier, sans fenêtre
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV   # GridSearchCV : recherche d'hyperparamètres par validation croisée
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve,
                             classification_report, confusion_matrix)

NOMS_CLASSES = ["fonctionnement", "panne"]    # index 0 -> "fonctionnement", index 1 -> "panne"


# --- Génération du jeu de données (fourni par l'énoncé) --------------------
def generer_donnees(n=3000, seed=42):
    rng = np.random.default_rng(seed)          # graine fixe -> reproductibilité
    df = pd.DataFrame({
        "temperature_air": rng.normal(300, 2, n),          # mesures de capteurs simulées
        "temperature_process": rng.normal(310, 1.5, n),
        "vitesse_rotation": rng.normal(1500, 150, n),
        "couple": rng.normal(40, 10, n),
        "usure_outil_min": rng.integers(0, 250, n),
        "vibration": rng.gamma(2.0, 1.0, n),               # loi gamma : toujours positive, asymétrique (réaliste pour une vibration)
    })
    # Score de risque caché : usure, couple, vibration et écart de température augmentent le risque
    score = (0.02 * df["usure_outil_min"]
             + 0.06 * df["couple"]
             + 0.5 * df["vibration"]
             + 0.4 * (df["temperature_process"] - df["temperature_air"] - 10)
             + rng.normal(0, 0.8, n))
    df["panne"] = (score > score.quantile(0.90)).astype(int)   # 10 % de pannes -> classe rare (déséquilibre)
    return df


# --- ÉTAPE 1 : Acquisition et partitionnement ------------------------------
def charger_et_separer():
    df = generer_donnees()                      # acquisition des données
    X = df.drop(columns="panne")                # variables explicatives (capteurs)
    y = df["panne"]                             # cible : 1 = panne
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,                         # 25 % réservés pour le test final
        stratify=y,                             # garde ~10 % de pannes dans train ET test (essentiel avec une classe rare)
        random_state=42,                        # séparation reproductible
    )
    return X_train, X_test, y_train, y_test, list(X.columns), NOMS_CLASSES


# --- ÉTAPE 2 : Comparaison de deux modèles par GridSearchCV ----------------
def optimiser_modeles(X_train, y_train):
    # Dictionnaire : nom -> (pipeline, grille d'hyperparamètres à tester)
    configs = {
        "Régression logistique": (
            Pipeline([
                ("scaler", StandardScaler()),   # indispensable pour la régression logistique (sensible aux échelles)
                ("clf", LogisticRegression(
                    max_iter=5000,              # assez d'itérations pour que l'optimisation converge
                    class_weight="balanced",    # donne plus de poids à la classe rare "panne" pour qu'elle ne soit pas ignorée
                    random_state=42)),
            ]),
            {"clf__C": [0.01, 0.1, 1, 10]},     # C = inverse de la régularisation ; "clf__" désigne l'étape "clf" du pipeline
        ),
        "Random Forest": (
            Pipeline([
                ("scaler", StandardScaler()),   # gardé pour avoir deux pipelines de même structure (sans effet sur les arbres)
                ("clf", RandomForestClassifier(
                    class_weight="balanced",    # même raison : compenser le déséquilibre des classes
                    random_state=42)),
            ]),
            {"clf__n_estimators": [100, 200],   # nombre d'arbres
             "clf__max_depth": [None, 5, 10]},  # profondeur max : limiter la profondeur réduit le sur-apprentissage
        ),
    }
    resultats = {}
    for nom, (pipe, grille) in configs.items():
        print(f"Optimisation : {nom} ...")
        gs = GridSearchCV(
            pipe, grille,
            cv=5,                  # validation croisée à 5 plis pour chaque combinaison
            scoring="roc_auc",     # AUC-ROC : adaptée aux classes déséquilibrées (l'accuracy serait trompeuse)
            n_jobs=-1,             # utilise tous les coeurs du processeur pour aller plus vite
        )
        gs.fit(X_train, y_train)   # teste toutes les combinaisons, puis ré-entraîne la meilleure sur tout le train
        resultats[nom] = gs
    return resultats


# --- ÉTAPE 3 : Évaluation finale sur le jeu de test ------------------------
def evaluer(resultats, X_test, y_test):
    scores_test = {}
    plt.figure(figsize=(6, 5))
    for nom, gs in resultats.items():
        proba = gs.best_estimator_.predict_proba(X_test)[:, 1]   # probabilité de panne (l'AUC a besoin de probabilités, pas de classes)
        auc = roc_auc_score(y_test, proba)                       # AUC : 1 = parfait, 0.5 = hasard
        scores_test[nom] = auc
        fpr, tpr, _ = roc_curve(y_test, proba)                   # points de la courbe ROC pour tous les seuils possibles
        plt.plot(fpr, tpr, label=f"{nom} (AUC={auc:.3f})")
        print(f"\n[{nom}]")
        print("  Meilleurs paramètres :", gs.best_params_)
        print(f"  AUC-ROC (CV)   : {gs.best_score_:.4f}")         # score moyen en validation croisée (sur le train)
        print(f"  AUC-ROC (test) : {auc:.4f}")                    # score sur données jamais vues
        # Si CV et test sont proches, le modèle généralise bien (pas de sur-apprentissage marqué).
    plt.plot([0, 1], [0, 1], "k--", label="Hasard")               # diagonale = modèle aléatoire (référence)
    plt.xlabel("Taux de fausses alertes")
    plt.ylabel("Taux de pannes détectées (rappel)")
    plt.title("Courbes ROC - maintenance prédictive")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc.png", dpi=120)
    plt.close()
    meilleur = max(scores_test, key=scores_test.get)              # nom du modèle ayant la plus grande AUC test
    print(f"\nMeilleur modèle : {meilleur}")
    return meilleur


# --- ÉTAPE 4 : Interprétation du meilleur modèle ---------------------------
def interpreter(meilleur_pipeline, X_test, y_test, noms_variables, noms_classes):
    y_pred = meilleur_pipeline.predict(X_test)        # classes prédites avec le seuil par défaut de 0.5
    print("\nMatrice de confusion :\n", confusion_matrix(y_test, y_pred))   # visualise les types d'erreurs
    print("\n--- Rapport de classification ---")
    print(classification_report(y_test, y_pred, target_names=noms_classes))  # précision / rappel / F1 par classe

    clf = meilleur_pipeline.named_steps["clf"]        # extrait le modèle final du pipeline
    if hasattr(clf, "feature_importances_"):          # cas des modèles à arbres (Random Forest)
        poids = clf.feature_importances_
        libelle = "importance"
    else:                                             # cas de la régression logistique
        poids = np.abs(clf.coef_[0])                  # |coefficient| : poids de la variable (valide car variables standardisées)
        libelle = "|coefficient|"
    indices = np.argsort(poids)[::-1][:3]             # indices des 3 plus gros poids, du plus grand au plus petit
    print(f"--- Top 3 des variables ({libelle}) ---")
    for rang, idx in enumerate(indices, start=1):
        print(f"{rang}. {noms_variables[idx]} ({poids[idx]:.4f})")


# --- ÉTAPE 5 : Mini-application --------------------------------------------
def sauvegarder_modele(modele, chemin="modele_maintenance.joblib"):
    joblib.dump(modele, chemin)     # on sauvegarde le PIPELINE entier (scaler + modèle) : en production, plus besoin de re-préparer les données à la main
    print(f"\nModèle sauvegardé dans {chemin}")


def predire_machine(chemin, mesures_df, seuil_panne=0.3):
    modele = joblib.load(chemin)                                  # recharge le pipeline sans ré-entraîner
    proba_panne = float(modele.predict_proba(mesures_df)[0, 1])   # probabilité de panne pour l'unique ligne fournie
    decision = ("ALERTE : maintenance préventive"
                if proba_panne >= seuil_panne                     # seuil 0.3 < 0.5 : on préfère une fausse alerte à une panne manquée
                else "Fonctionnement normal")
    return {"proba_panne": round(proba_panne, 4), "decision": decision}


# --- Orchestration de toute l'étude de cas ---------------------------------
def run_case_study():
    X_train, X_test, y_train, y_test, noms_vars, noms_classes = charger_et_separer()
    resultats = optimiser_modeles(X_train, y_train)                 # optimisation SUR LE TRAIN uniquement
    meilleur_nom = evaluer(resultats, X_test, y_test)               # le test n'intervient qu'ici, pour la comparaison finale
    meilleur = resultats[meilleur_nom].best_estimator_              # pipeline gagnant, déjà entraîné
    interpreter(meilleur, X_test, y_test, noms_vars, noms_classes)
    sauvegarder_modele(meilleur)

    machine = X_test.iloc[[0]]  # double crochets -> DataFrame d'une seule ligne (conserve les noms de colonnes attendus par le pipeline)
    print("\nPrédiction mini-app :", predire_machine("modele_maintenance.joblib", machine))
    print("Vraie classe :", noms_classes[y_test.iloc[0]])           # comparaison avec la réalité


# Ce bloc ne s'exécute que si le fichier est lancé directement (pas s'il est importé ailleurs)
if __name__ == "__main__":
    run_case_study()

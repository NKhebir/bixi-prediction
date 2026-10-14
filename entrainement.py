"""Entraîner, comparer et évaluer les modèles."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # dessine les graphiques dans des fichiers, sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bixi.caracteristiques import COLONNES_MODELE, NOMS_FRANCAIS, ajouter_caracteristiques
from config import DATE_COUPURE, DOSSIER_MODELES, DOSSIER_PREPARE, DOSSIER_RESULTATS


def evaluer(vrai, prevu):
    return {
        "MAE": float(mean_absolute_error(vrai, prevu)),          # erreur moyenne, en vélos
        "RMSE": float(np.sqrt(mean_squared_error(vrai, prevu))),  # punit plus les grosses erreurs
        "R2": float(r2_score(vrai, prevu)),                       # 1 = parfait, 0 = pas mieux qu'une moyenne
    }


def charger_donnees():
    grille = pd.read_parquet(DOSSIER_PREPARE / "departs_par_heure.parquet")
    meteo = pd.read_parquet(DOSSIER_PREPARE / "meteo.parquet")
    stations = pd.read_csv(DOSSIER_PREPARE / "stations.csv")
    return grille.merge(meteo, on="heure", how="left"), stations


def entrainer():
    donnees, stations_df = charger_donnees()
    stations = stations_df["station"].tolist()

    # 1. Séparer le passé (entraînement) du futur (test) : le modèle ne voit jamais le test.
    coupure = pd.Timestamp(DATE_COUPURE)
    entrainement = donnees[donnees["heure"] < coupure]
    test = donnees[donnees["heure"] >= coupure]
    print(f"Entraînement : {len(entrainement):,} lignes (avant le {DATE_COUPURE})")
    print(f"Test         : {len(test):,} lignes (à partir du {DATE_COUPURE})")

    # La popularité est calculée SEULEMENT sur l'entraînement, sinon on « tricherait ».
    popularite = entrainement.groupby("station")["departs"].mean().to_dict()
    entrainement = ajouter_caracteristiques(entrainement, stations, popularite)
    test = ajouter_caracteristiques(test, stations, popularite)
    X_ent, y_ent = entrainement[COLONNES_MODELE], entrainement["departs"]
    X_test, y_test = test[COLONNES_MODELE], test["departs"]

    resultats, predictions, modeles = {}, {}, {}

    # 2. Référence « bête » : la moyenne habituelle pour cette station, ce jour et cette heure.
    moyennes = entrainement.groupby(["station", "jour_semaine", "heure_jour"])["departs"].mean()
    cles = pd.MultiIndex.from_frame(test[["station", "jour_semaine", "heure_jour"]])
    p = moyennes.reindex(cles).fillna(y_ent.mean()).to_numpy()
    predictions["Moyenne historique"] = p
    resultats["Moyenne historique"] = evaluer(y_test, p)
    print(f"\nMoyenne historique : MAE = {resultats['Moyenne historique']['MAE']:.3f}")

    # 3. Forêt aléatoire : 100 arbres de décision qui votent ensemble.
    print("Entraînement de la forêt aléatoire...")
    foret = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=10, max_samples=0.5, n_jobs=-1, random_state=42
    )
    foret.fit(X_ent, y_ent)
    modeles["Forêt aléatoire"] = foret
    predictions["Forêt aléatoire"] = foret.predict(X_test)
    resultats["Forêt aléatoire"] = evaluer(y_test, predictions["Forêt aléatoire"])
    print(f"Forêt aléatoire : MAE = {resultats['Forêt aléatoire']['MAE']:.3f}")

    # 4. Gradient boosting : des centaines de petits arbres, chacun corrige les erreurs du précédent.
    #    La perte « poisson » est faite pour prédire des nombres qui comptent quelque chose.
    print("Entraînement du gradient boosting...")
    boosting = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=600,
        learning_rate=0.08,
        max_leaf_nodes=63,
        categorical_features=[COLONNES_MODELE.index("station_code")],
        random_state=42,
    )
    boosting.fit(X_ent, y_ent)
    modeles["Gradient boosting"] = boosting
    predictions["Gradient boosting"] = boosting.predict(X_test)
    resultats["Gradient boosting"] = evaluer(y_test, predictions["Gradient boosting"])
    print(f"Gradient boosting : MAE = {resultats['Gradient boosting']['MAE']:.3f}")

    # 5. On garde le meilleur vrai modèle (la plus petite erreur moyenne).
    meilleur = min(modeles, key=lambda nom: resultats[nom]["MAE"])
    print(f"\nMeilleur modèle : {meilleur}")

    DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "modele": modeles[meilleur],
            "nom_modele": meilleur,
            "stations": stations,
            "popularite": popularite,
            "date_coupure": DATE_COUPURE,
            "resultats": resultats,
        },
        DOSSIER_MODELES / "modele_bixi.joblib",
        compress=3,
    )

    test = test.assign(prevu=predictions[meilleur])
    produire_rapport(test, resultats, meilleur, modeles[meilleur], X_test, y_test)


def produire_rapport(test, resultats, meilleur, modele, X_test, y_test):
    """Graphiques et tableau des résultats, dans le dossier resultats/."""
    DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)
    reference = resultats["Moyenne historique"]["MAE"]
    gain = (1 - resultats[meilleur]["MAE"] / reference) * 100
    moyenne_test = float(y_test.mean())

    # Graphique 1 : comparaison des erreurs
    fig, ax = plt.subplots(figsize=(7, 4))
    noms = list(resultats)
    couleurs = ["#9aa5b1" if n != meilleur else "#e4002b" for n in noms]
    ax.bar(noms, [resultats[n]["MAE"] for n in noms], color=couleurs)
    ax.set_ylabel("Erreur moyenne (vélos par heure)")
    ax.set_title("Erreur des modèles sur la période de test (plus bas = mieux)")
    fig.tight_layout()
    fig.savefig(DOSSIER_RESULTATS / "comparaison_modeles.png", dpi=130)
    plt.close(fig)

    # Graphique 2 : total par jour, réel vs prévu, pour les 50 stations
    par_jour = test.groupby(test["heure"].dt.date)[["departs", "prevu"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(par_jour.index, par_jour["departs"], label="Réel", color="#1f2933")
    ax.plot(par_jour.index, par_jour["prevu"], label="Prévu", color="#e4002b", alpha=0.85)
    ax.set_ylabel("Départs par jour (50 stations)")
    ax.set_title("Départs quotidiens : réel vs prévu (période de test)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DOSSIER_RESULTATS / "reel_vs_prevu_par_jour.png", dpi=130)
    plt.close(fig)

    # Graphique 3 : une semaine heure par heure, pour la station la plus utilisée
    station = test.groupby("station")["departs"].sum().idxmax()
    lundi = test.loc[test["heure"].dt.dayofweek == 0, "heure"].dt.normalize().min()
    semaine = test[(test["station"] == station) & (test["heure"] >= lundi)
                   & (test["heure"] < lundi + pd.Timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(semaine["heure"], semaine["departs"], label="Réel", color="#1f2933")
    ax.plot(semaine["heure"], semaine["prevu"], label="Prévu", color="#e4002b", alpha=0.85)
    ax.set_ylabel("Départs par heure")
    ax.set_title(f"Une semaine à la station « {station} »")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DOSSIER_RESULTATS / "semaine_station.png", dpi=130)
    plt.close(fig)

    # Graphique 4 : ce qui compte le plus pour le modèle.
    # On mélange une caractéristique à la fois : si l'erreur augmente beaucoup, elle est importante.
    echantillon = X_test.sample(min(20_000, len(X_test)), random_state=0)
    importance = permutation_importance(
        modele, echantillon, y_test.loc[echantillon.index],
        n_repeats=3, random_state=0, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    imp = pd.Series(importance.importances_mean, index=[NOMS_FRANCAIS[c] for c in X_test.columns])
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color="#e4002b")
    ax.set_xlabel("Hausse de l'erreur quand on « brouille » cette information")
    ax.set_title("Importance des caractéristiques")
    fig.tight_layout()
    fig.savefig(DOSSIER_RESULTATS / "importance_caracteristiques.png", dpi=130)
    plt.close(fig)

    # Fichier de chiffres (lu par l'application)
    metriques = {
        "meilleur_modele": meilleur,
        "gain_vs_moyenne_pourcent": gain,
        "moyenne_departs_par_heure_test": moyenne_test,
        "periode_test_debut": str(test["heure"].min().date()),
        "periode_test_fin": str(test["heure"].max().date()),
        "lignes_test": int(len(test)),
        "resultats": resultats,
        "importance": imp.sort_values(ascending=False).to_dict(),
        "station_graphique": station,
    }
    (DOSSIER_RESULTATS / "metriques.json").write_text(
        json.dumps(metriques, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Rapport lisible sur GitHub
    lignes = [
        "# Résultats",
        "",
        f"Période de test : du {metriques['periode_test_debut']} au {metriques['periode_test_fin']} "
        f"({len(test):,} heures × stations), jamais vue pendant l'entraînement.",
        "",
        "| Modèle | Erreur moyenne (MAE) | RMSE | R² |",
        "|---|---|---|---|",
    ]
    for nom, r in resultats.items():
        gras = "**" if nom == meilleur else ""
        lignes.append(f"| {gras}{nom}{gras} | {r['MAE']:.2f} | {r['RMSE']:.2f} | {r['R2']:.3f} |")
    lignes += [
        "",
        f"Le meilleur modèle (**{meilleur}**) se trompe en moyenne de "
        f"**{resultats[meilleur]['MAE']:.2f} départ(s) par heure** (moyenne réelle : "
        f"{moyenne_test:.2f} départs par heure), soit **{gain:.0f} % d'erreur en moins** "
        "que la simple moyenne historique.",
        "",
        "Les trois informations les plus utiles au modèle : "
        + ", ".join(imp.sort_values(ascending=False).index[:3]) + ".",
        "",
        "![Comparaison des modèles](comparaison_modeles.png)",
        "![Réel vs prévu par jour](reel_vs_prevu_par_jour.png)",
        "![Une semaine](semaine_station.png)",
        "![Importance](importance_caracteristiques.png)",
        "",
    ]
    (DOSSIER_RESULTATS / "RESULTATS.md").write_text("\n".join(lignes), encoding="utf-8")
    print("\n".join(lignes[:-7]))
    print(f"\nGraphiques et rapport enregistrés dans {DOSSIER_RESULTATS}/")

"""Utiliser le modèle entraîné pour prédire une journée."""
import joblib
import pandas as pd

from bixi.caracteristiques import COLONNES_MODELE, METEO_PAR_DEFAUT, ajouter_caracteristiques
from config import DOSSIER_MODELES


def charger_modele():
    return joblib.load(DOSSIER_MODELES / "modele_bixi.joblib")


def meteo_constante(date, temperature, precipitation, vent):
    """Même météo pour les 24 heures (quand on choisit la météo à la main)."""
    heures = pd.date_range(pd.Timestamp(date), periods=24, freq="h")
    return pd.DataFrame(
        {"heure": heures, "temperature": temperature, "precipitation": precipitation, "vent": vent}
    )


def predire_journee(paquet, station, date, meteo):
    """Retourne les départs prévus pour chaque heure de la journée à cette station."""
    if station not in paquet["stations"]:
        raise ValueError(f"Station inconnue : {station}")
    heures = pd.DataFrame(
        {"heure": pd.date_range(pd.Timestamp(date), periods=24, freq="h"), "station": station}
    )
    df = heures.merge(meteo, on="heure", how="left")
    for colonne, valeur in METEO_PAR_DEFAUT.items():
        if colonne not in df:
            df[colonne] = valeur
    df = ajouter_caracteristiques(df, paquet["stations"], paquet["popularite"])
    prevu = paquet["modele"].predict(df[COLONNES_MODELE]).clip(min=0)
    return pd.DataFrame(
        {
            "heure": df["heure"],
            "departs_prevus": prevu.round(1),
            "temperature": df["temperature"],
            "precipitation": df["precipitation"],
        }
    )

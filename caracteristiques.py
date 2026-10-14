"""Les « caractéristiques » : les indices que l'on donne au modèle pour qu'il prédise."""
import holidays
import pandas as pd

COLONNES_MODELE = [
    "station_code",   # quelle station (un numéro)
    "popularite",     # nombre moyen de départs par heure de cette station (dans l'entraînement)
    "heure_jour",     # 0 à 23
    "jour_semaine",   # 0 = lundi ... 6 = dimanche
    "mois",
    "annee",
    "weekend",        # 1 si samedi ou dimanche
    "ferie",          # 1 si jour férié au Québec
    "temperature",
    "precipitation",
    "vent",
]

NOMS_FRANCAIS = {
    "station_code": "Station",
    "popularite": "Popularité de la station",
    "heure_jour": "Heure de la journée",
    "jour_semaine": "Jour de la semaine",
    "mois": "Mois",
    "annee": "Année",
    "weekend": "Week-end",
    "ferie": "Jour férié",
    "temperature": "Température",
    "precipitation": "Pluie",
    "vent": "Vent",
}

METEO_PAR_DEFAUT = {"temperature": 18.0, "precipitation": 0.0, "vent": 12.0}


def jours_feries(annees):
    return list(holidays.Canada(subdiv="QC", years=list(annees)).keys())


def ajouter_caracteristiques(df, stations, popularite):
    """df doit contenir les colonnes : station, heure, temperature, precipitation, vent."""
    df = df.copy()
    h = df["heure"]
    df["heure_jour"] = h.dt.hour
    df["jour_semaine"] = h.dt.dayofweek
    df["mois"] = h.dt.month
    df["annee"] = h.dt.year
    df["weekend"] = (df["jour_semaine"] >= 5).astype(int)
    df["ferie"] = h.dt.date.isin(jours_feries(sorted(h.dt.year.unique()))).astype(int)
    df["station_code"] = df["station"].map({s: i for i, s in enumerate(stations)})
    df["popularite"] = df["station"].map(popularite)
    for colonne, valeur in METEO_PAR_DEFAUT.items():
        df[colonne] = df[colonne].fillna(valeur)
    return df

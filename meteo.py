"""Météo heure par heure à Montréal, grâce à l'API gratuite Open-Meteo (sans clé)."""
import pandas as pd
import requests

from config import FUSEAU, LATITUDE, LONGITUDE

URL_HISTORIQUE = "https://archive-api.open-meteo.com/v1/archive"
URL_PREVISIONS = "https://api.open-meteo.com/v1/forecast"
VARIABLES = "temperature_2m,precipitation,wind_speed_10m"


def _en_tableau(donnees):
    h = donnees["hourly"]
    df = pd.DataFrame(
        {
            "heure": pd.to_datetime(h["time"]),
            "temperature": h["temperature_2m"],   # °C
            "precipitation": h["precipitation"],  # mm de pluie dans l'heure
            "vent": h["wind_speed_10m"],          # km/h
        }
    )
    # Au changement d'heure d'automne, une heure apparaît deux fois : on n'en garde qu'une.
    df = df.drop_duplicates("heure").sort_values("heure")
    colonnes = ["temperature", "precipitation", "vent"]
    df[colonnes] = df[colonnes].interpolate(limit_direction="both")  # bouche les rares trous
    return df.reset_index(drop=True)


def _appeler(url, parametres):
    parametres = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": VARIABLES,
        "timezone": FUSEAU,
        **parametres,
    }
    reponse = requests.get(url, params=parametres, timeout=60)
    reponse.raise_for_status()
    return _en_tableau(reponse.json())


def meteo_historique(debut, fin):
    """Météo réelle passée, par exemple de '2024-01-01' à '2025-12-31'."""
    return _appeler(URL_HISTORIQUE, {"start_date": debut, "end_date": fin})


def meteo_previsions():
    """Prévisions météo pour les 16 prochains jours."""
    return _appeler(URL_PREVISIONS, {"forecast_days": 16})

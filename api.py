"""API web des prédictions.  Lancer avec :  uvicorn api:app --reload
Puis ouvrir http://127.0.0.1:8000/docs pour la tester dans le navigateur."""
from datetime import date

from fastapi import FastAPI, HTTPException

from bixi.meteo import meteo_previsions
from bixi.prediction import charger_modele, meteo_constante, predire_journee

app = FastAPI(title="API de prédiction BIXI", description="Départs prévus par station et par heure.")
paquet = charger_modele()


@app.get("/stations")
def lister_stations():
    return {"stations": paquet["stations"]}


@app.get("/prediction")
def prediction(station: str, jour: date, temperature: float | None = None,
               precipitation: float = 0.0, vent: float = 12.0):
    """Si la température n'est pas donnée, on utilise les prévisions météo réelles."""
    if station not in paquet["stations"]:
        raise HTTPException(404, "Station inconnue. Voir /stations pour la liste.")
    if temperature is not None:
        meteo = meteo_constante(jour, temperature, precipitation, vent)
    else:
        try:
            previsions = meteo_previsions()
        except Exception:
            raise HTTPException(503, "Prévisions météo indisponibles : donne une température.")
        meteo = previsions[previsions["heure"].dt.date == jour]
        if len(meteo) < 24:
            raise HTTPException(400, "Pas de prévisions pour cette date (max. 16 jours) : "
                                     "donne une température.")
    resultat = predire_journee(paquet, station, jour, meteo)
    return {
        "station": station,
        "jour": str(jour),
        "total_prevu": round(float(resultat["departs_prevus"].sum()), 1),
        "par_heure": [
            {"heure": h.strftime("%H:%M"), "departs_prevus": float(d)}
            for h, d in zip(resultat["heure"], resultat["departs_prevus"])
        ],
    }

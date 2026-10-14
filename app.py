"""Interface web.  Lancer avec :  streamlit run app.py"""
import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from bixi.meteo import meteo_previsions
from bixi.prediction import charger_modele, meteo_constante, predire_journee
from config import DOSSIER_MODELES, DOSSIER_PREPARE, DOSSIER_RESULTATS

st.set_page_config(page_title="Prédiction BIXI", page_icon="🚲", layout="wide")
st.title("🚲 Prédiction de la demande BIXI")
st.caption("Combien de vélos partiront de cette station, heure par heure ? "
           "Modèle entraîné sur les vrais trajets BIXI 2024-2025 et la météo de Montréal.")

if not (DOSSIER_MODELES / "modele_bixi.joblib").exists():
    st.error("Aucun modèle trouvé. Lance d'abord : python 1_telecharger.py, "
             "python 2_preparer.py puis python 3_entrainer.py")
    st.stop()


@st.cache_resource
def get_modele():
    return charger_modele()


@st.cache_data(ttl=3600)  # on redemande les prévisions au maximum une fois par heure
def get_previsions():
    try:
        return meteo_previsions()
    except Exception:
        return None


paquet = get_modele()
stations = pd.read_csv(DOSSIER_PREPARE / "stations.csv")
onglet_prediction, onglet_performance = st.tabs(["Prédire", "Performance du modèle"])

with onglet_prediction:
    col1, col2 = st.columns([1, 1])
    with col1:
        station = st.selectbox("Station", paquet["stations"])
        jour = st.date_input("Date", value=date.today() + timedelta(days=1))

        previsions = get_previsions()
        meteo_du_jour = None
        if previsions is not None:
            meteo_du_jour = previsions[previsions["heure"].dt.date == jour]
            if len(meteo_du_jour) < 24:
                meteo_du_jour = None

        options = ["Choisir la météo moi-même"]
        if meteo_du_jour is not None:
            options.insert(0, "Prévisions météo réelles (Open-Meteo)")
        mode = st.radio("Météo", options)

        if mode.startswith("Prévisions"):
            meteo = meteo_du_jour
            st.caption(f"Température prévue : {meteo['temperature'].min():.0f} à "
                       f"{meteo['temperature'].max():.0f} °C, pluie : {meteo['precipitation'].sum():.1f} mm")
        else:
            temperature = st.slider("Température (°C)", -5, 35, 20)
            pluie = st.slider("Pluie (mm par heure)", 0.0, 10.0, 0.0, step=0.5)
            vent = st.slider("Vent (km/h)", 0, 60, 12)
            meteo = meteo_constante(jour, temperature, pluie, vent)

    with col2:
        if "lat" in stations:
            carte = stations.dropna(subset=["lat", "lon"])
            st.map(carte.rename(columns={"lat": "latitude", "lon": "longitude"}), size=40)

    resultat = predire_journee(paquet, station, jour, meteo)
    pointe = resultat.loc[resultat["departs_prevus"].idxmax()]
    m1, m2, m3 = st.columns(3)
    m1.metric("Départs prévus dans la journée", f"{resultat['departs_prevus'].sum():.0f}")
    m2.metric("Heure de pointe", f"{pointe['heure']:%H} h")
    m3.metric("Départs à l'heure de pointe", f"{pointe['departs_prevus']:.0f}")

    graphique = resultat.assign(Heure=resultat["heure"].dt.strftime("%H h")).set_index("Heure")
    st.bar_chart(graphique["departs_prevus"], y_label="Départs prévus", color="#e4002b")
    if jour.month not in range(4, 12):
        st.warning("Le modèle est entraîné d'avril à novembre : en hiver, la prédiction est peu fiable.")

with onglet_performance:
    fichier = DOSSIER_RESULTATS / "metriques.json"
    if fichier.exists():
        m = json.loads(fichier.read_text(encoding="utf-8"))
        meilleur = m["meilleur_modele"]
        st.subheader(f"Meilleur modèle : {meilleur}")
        st.write(
            f"Testé sur des données jamais vues ({m['periode_test_debut']} au {m['periode_test_fin']}). "
            f"Erreur moyenne : **{m['resultats'][meilleur]['MAE']:.2f} départ(s) par heure**, "
            f"soit **{m['gain_vs_moyenne_pourcent']:.0f} % de moins** que la moyenne historique."
        )
        tableau = pd.DataFrame(m["resultats"]).T.round(3)
        tableau.columns = ["Erreur moyenne (MAE)", "RMSE", "R²"]
        st.dataframe(tableau)
        for image in ["comparaison_modeles", "reel_vs_prevu_par_jour", "semaine_station",
                      "importance_caracteristiques"]:
            chemin = DOSSIER_RESULTATS / f"{image}.png"
            if chemin.exists():
                st.image(str(chemin))

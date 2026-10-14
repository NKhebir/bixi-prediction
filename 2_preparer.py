"""Étape 2 : compter les départs par station et par heure.  python 2_preparer.py"""
import pandas as pd

from bixi.preparation import construire_grille, lire_un_zip
from config import DOSSIER_BRUT, DOSSIER_PREPARE, MOIS_SAISON, NB_STATIONS, URLS_BIXI

tous_departs, toutes_coords = [], []
for annee in URLS_BIXI:
    print(f"Lecture des trajets {annee}...")
    departs, coords = lire_un_zip(DOSSIER_BRUT / f"bixi_{annee}.zip")
    tous_departs.append(departs)
    if coords is not None:
        toutes_coords.append(coords)

departs = pd.concat(tous_departs).groupby(["station", "heure"], as_index=False)["departs"].sum()
print(f"Total : {departs['departs'].sum():,} trajets lus")

# On garde la saison principale et les stations les plus utilisées.
departs = departs[departs["heure"].dt.month.isin(list(MOIS_SAISON))]
totaux = departs.groupby("station")["departs"].sum().nlargest(NB_STATIONS)
departs = departs[departs["station"].isin(totaux.index)]

grille = construire_grille(departs)
DOSSIER_PREPARE.mkdir(parents=True, exist_ok=True)
grille.to_parquet(DOSSIER_PREPARE / "departs_par_heure.parquet", index=False)

stations = totaux.rename("total_departs").reset_index()
if toutes_coords:
    coords = pd.concat(toutes_coords).groupby(level=0).median()
    stations = stations.merge(coords, left_on="station", right_index=True, how="left")
stations.to_csv(DOSSIER_PREPARE / "stations.csv", index=False)

print(f"{len(grille):,} lignes (station × heure) pour {len(stations)} stations")
print(f"Station la plus utilisée : {stations.iloc[0]['station']} ({stations.iloc[0]['total_departs']:,} départs)")
print("Terminé. Prochaine étape : python 3_entrainer.py")

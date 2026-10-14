"""Étape 1 : télécharger les vraies données (trajets BIXI + météo).  python 1_telecharger.py"""
from bixi.meteo import meteo_historique
from bixi.telechargement import telecharger
from config import DOSSIER_BRUT, DOSSIER_PREPARE, URLS_BIXI

print("Téléchargement des trajets BIXI (plusieurs centaines de Mo, patience)...")
for annee, url in URLS_BIXI.items():
    telecharger(url, DOSSIER_BRUT / f"bixi_{annee}.zip")

print("Téléchargement de la météo heure par heure (Open-Meteo)...")
annees = sorted(URLS_BIXI)
meteo = meteo_historique(f"{annees[0]}-01-01", f"{annees[-1]}-12-31")
DOSSIER_PREPARE.mkdir(parents=True, exist_ok=True)
meteo.to_parquet(DOSSIER_PREPARE / "meteo.parquet", index=False)
print(f"  {len(meteo):,} heures de météo enregistrées")
print("Terminé. Prochaine étape : python 2_preparer.py")

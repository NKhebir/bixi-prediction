"""Réglages du projet : on change les choix ici, pas dans le code."""
from pathlib import Path

RACINE = Path(__file__).resolve().parent
DOSSIER_BRUT = RACINE / "data" / "brut"          # fichiers téléchargés tels quels
DOSSIER_PREPARE = RACINE / "data" / "prepare"    # données nettoyées, prêtes pour le modèle
DOSSIER_MODELES = RACINE / "modeles"             # le modèle entraîné
DOSSIER_RESULTATS = RACINE / "resultats"         # chiffres et graphiques de performance

# Données ouvertes de BIXI Montréal (licence CC-BY 4.0) : https://bixi.com/fr/donnees-ouvertes/
URLS_BIXI = {
    2024: "https://cdn.bixi.com/wp-content/uploads/2025/01/DonneesOuvertes2024_010203040506070809101112.zip",
    2025: "https://cdn.bixi.com/wp-content/uploads/2026/02/DonneesOuvertes2025_010203040506070809101112.zip",
}

MOIS_SAISON = range(4, 12)   # avril à novembre : la saison principale de BIXI
NB_STATIONS = 50             # on garde les 50 stations les plus utilisées
DATE_COUPURE = "2025-08-01"  # avant = entraînement, à partir de cette date = test

# Centre-ville de Montréal, pour la météo
LATITUDE, LONGITUDE = 45.5017, -73.5673
FUSEAU = "America/Toronto"   # même fuseau horaire que Montréal

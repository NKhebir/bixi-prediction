"""Transformer des millions de trajets en un tableau : station × heure → nombre de départs."""
import zipfile

import pandas as pd

from config import FUSEAU

# Colonnes du format BIXI actuel (2022 et après) que l'on garde, et leur nouveau nom.
COLONNES = {
    "STARTSTATIONNAME": "station",
    "STARTTIMEMS": "temps_ms",
    "STARTSTATIONLATITUDE": "lat",
    "STARTSTATIONLONGITUDE": "lon",
}


def _encodage(archive, nom):
    """Devine si le fichier est en UTF-8 ou en Latin-1 (important pour les accents)."""
    with archive.open(nom) as f:
        debut = f.read(200_000)
    debut = debut[: debut.rfind(b"\n")]  # on coupe à une fin de ligne pour ne pas couper une lettre
    try:
        debut.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _verifier_entete(archive, nom, encodage):
    """Lit seulement la première ligne pour vérifier les colonnes et deviner le séparateur."""
    with archive.open(nom) as f:
        entete = f.readline().decode(encodage).strip()
    separateur = ";" if entete.count(";") > entete.count(",") else ","
    colonnes = {c.strip().strip('"').upper() for c in entete.split(separateur)}
    if not {"STARTSTATIONNAME", "STARTTIMEMS"} <= colonnes:
        raise ValueError(
            f"{nom} n'a pas le format attendu. Colonnes trouvées : {sorted(colonnes)}. "
            "Ce projet utilise les fichiers BIXI de 2022 et après (STARTSTATIONNAME, STARTTIMEMS)."
        )
    return separateur


def _vers_heure_locale(temps):
    """Horodatage Unix (millisecondes) → heure de Montréal, arrondie à l'heure."""
    temps = pd.to_numeric(temps, errors="coerce")
    dates = pd.to_datetime(temps, unit="ms", utc=True, errors="coerce")
    if dates.dropna().dt.year.median() < 2000:  # sécurité : le temps était en secondes
        dates = pd.to_datetime(temps, unit="s", utc=True, errors="coerce")
    return dates.dt.tz_convert(FUSEAU).dt.tz_localize(None).dt.floor("h")


def lire_un_zip(chemin):
    """Lit un fichier ZIP de BIXI et compte les départs par station et par heure.

    Le fichier est trop gros pour être lu d'un coup : on le lit par morceaux
    d'un million de lignes, et on ne garde que les comptes.
    """
    comptes, coordonnees = [], []
    with zipfile.ZipFile(chemin) as archive:
        fichiers_csv = [
            n for n in archive.namelist()
            if n.lower().endswith(".csv") and "__macosx" not in n.lower()
        ]
        if not fichiers_csv:
            raise ValueError(f"Aucun fichier CSV trouvé dans {chemin.name}")

        for nom in fichiers_csv:
            encodage = _encodage(archive, nom)
            separateur = _verifier_entete(archive, nom, encodage)
            with archive.open(nom) as f:
                lecteur = pd.read_csv(
                    f,
                    sep=separateur,
                    encoding=encodage,
                    chunksize=1_000_000,
                    usecols=lambda c: c.strip().strip('"').upper() in COLONNES,
                    low_memory=False,
                )
                for i, morceau in enumerate(lecteur, start=1):
                    morceau = morceau.rename(columns=lambda c: COLONNES[c.strip().strip('"').upper()])
                    morceau["heure"] = _vers_heure_locale(morceau["temps_ms"])
                    morceau = morceau.dropna(subset=["station", "heure"])
                    comptes.append(morceau.groupby(["station", "heure"]).size())
                    if {"lat", "lon"} <= set(morceau.columns):
                        coordonnees.append(morceau.groupby("station")[["lat", "lon"]].median())
                    print(f"  {nom} : {i} million(s) de lignes lues")

    departs = pd.concat(comptes).groupby(level=[0, 1]).sum().rename("departs").reset_index()
    coords = pd.concat(coordonnees).groupby(level=0).median() if coordonnees else None
    return departs, coords


def construire_grille(departs):
    """Ajoute les heures avec ZÉRO départ, qui n'apparaissent pas dans les trajets.

    Sans ça, le modèle ne verrait jamais une heure calme et croirait que chaque heure
    a au moins un départ. On garde seulement les jours où la station était ouverte
    (au moins un départ dans la journée).
    """
    departs = departs.copy()
    departs["jour"] = departs["heure"].dt.normalize()
    jours_ouverts = departs[["station", "jour"]].drop_duplicates()
    heures = pd.DataFrame({"h": range(24)})
    grille = jours_ouverts.merge(heures, how="cross")
    grille["heure"] = grille["jour"] + pd.to_timedelta(grille["h"], unit="h")
    grille = grille.merge(departs[["station", "heure", "departs"]], on=["station", "heure"], how="left")
    grille["departs"] = grille["departs"].fillna(0).astype(int)
    return grille[["station", "heure", "departs"]].sort_values(["station", "heure"]).reset_index(drop=True)

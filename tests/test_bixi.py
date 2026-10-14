"""Tests automatiques.  Lancer avec :  pytest"""
import zipfile

import pandas as pd

from bixi.caracteristiques import ajouter_caracteristiques
from bixi.preparation import construire_grille, lire_un_zip


def test_jours_feries_et_weekend():
    df = pd.DataFrame({
        "station": "A",
        "heure": pd.to_datetime(["2024-06-24 08:00", "2024-06-25 08:00", "2024-06-29 08:00"]),
        "temperature": 20.0, "precipitation": 0.0, "vent": 10.0,
    })
    r = ajouter_caracteristiques(df, ["A"], {"A": 1.0})
    assert r["ferie"].tolist() == [1, 0, 0]     # 24 juin = Fête nationale du Québec
    assert r["weekend"].tolist() == [0, 0, 1]   # 29 juin 2024 = samedi


def test_grille_ajoute_les_heures_a_zero():
    departs = pd.DataFrame({
        "station": ["A", "A"],
        "heure": pd.to_datetime(["2024-07-01 08:00", "2024-07-01 17:00"]),
        "departs": [5, 3],
    })
    grille = construire_grille(departs)
    assert len(grille) == 24                    # toutes les heures de la journée
    assert grille["departs"].sum() == 8
    assert (grille["departs"] == 0).sum() == 22


def test_lecture_zip_et_heure_de_montreal(tmp_path):
    # 12 h 00 UTC le 1er juillet = 8 h 00 à Montréal (heure d'été)
    ms = int(pd.Timestamp("2024-07-01 12:00", tz="UTC").timestamp() * 1000)
    csv = ("STARTSTATIONNAME,STARTSTATIONLATITUDE,STARTSTATIONLONGITUDE,STARTTIMEMS,ENDTIMEMS\n"
           f"Métro Mont-Royal,45.52,-73.58,{ms},{ms + 600000}\n"
           f"Métro Mont-Royal,45.52,-73.58,{ms + 60000},{ms + 700000}\n")
    chemin = tmp_path / "test.zip"
    with zipfile.ZipFile(chemin, "w") as z:
        z.writestr("trajets.csv", csv)
    departs, coords = lire_un_zip(chemin)
    assert len(departs) == 1
    assert departs.iloc[0]["heure"] == pd.Timestamp("2024-07-01 08:00")
    assert departs.iloc[0]["departs"] == 2
    assert departs.iloc[0]["station"] == "Métro Mont-Royal"   # les accents sont bien lus
    assert abs(coords.loc["Métro Mont-Royal", "lat"] - 45.52) < 1e-9


def test_mauvais_format_donne_un_message_clair(tmp_path):
    chemin = tmp_path / "ancien.zip"
    with zipfile.ZipFile(chemin, "w") as z:
        z.writestr("2019.csv", "start_date,start_station_code\n2019-05-01 08:00,6001\n")
    try:
        lire_un_zip(chemin)
        assert False, "une erreur était attendue"
    except ValueError as e:
        assert "START_STATION_CODE" in str(e)

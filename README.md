# 🚲 Prédiction de la demande BIXI à Montréal

Un modèle de **machine learning** qui prédit combien de vélos partiront d'une station BIXI,
**heure par heure**, selon le jour, l'heure et la météo. Il est entraîné sur les **vrais
trajets BIXI 2024-2025** et la **vraie météo** de Montréal.

➡️ **Résultats complets : [resultats/RESULTATS.md](resultats/RESULTATS.md)** (générés automatiquement par l'étape 3)

## Ce que fait le projet

1. **Télécharge les vraies données** : des millions de trajets publiés par BIXI, et la météo heure par heure.
2. **Les prépare** : compte les départs par station et par heure, en ajoutant les heures sans aucun départ.
3. **Entraîne et compare trois approches**, testées sur des mois que le modèle n'a jamais vus :
   - une **moyenne historique** (la référence à battre) ;
   - une **forêt aléatoire** (*random forest*) ;
   - un **gradient boosting** avec perte de Poisson, adaptée à la prédiction de comptes.
4. **Explique ses choix** : graphique de l'importance de chaque information (heure, météo, jour férié…).
5. **Se consulte de deux façons** : une application web Streamlit (avec les prévisions météo réelles) et une API FastAPI.

## Méthode

| Étape | Choix | Pourquoi |
|---|---|---|
| Données | 50 stations les plus utilisées, avril à novembre | La saison principale, avec assez de trajets par station |
| Heures sans départ | Ajoutées avec la valeur 0 | Sinon le modèle ne verrait jamais une heure calme |
| Séparation | Entraînement avant le 1er août 2025, test après | On teste sur le **futur**, comme dans la vraie vie |
| Caractéristiques | Station, popularité, heure, jour, mois, année, week-end, jour férié (Québec), température, pluie, vent | Ce qui influence l'envie de prendre un vélo |
| Fuite de données | La popularité des stations est calculée seulement sur l'entraînement | Pour ne pas « tricher » avec le test |
| Mesures | MAE, RMSE, R² | L'erreur moyenne en vélos est facile à interpréter |

## Structure

```
1_telecharger.py   Étape 1 : télécharge les trajets BIXI et la météo
2_preparer.py      Étape 2 : compte les départs par station et par heure
3_entrainer.py     Étape 3 : entraîne, compare, évalue et produit les graphiques
app.py             Application web (Streamlit)
api.py             API web (FastAPI)
config.py          Tous les réglages (années, nombre de stations, date de coupure)
bixi/              Le code : téléchargement, météo, préparation, caractéristiques, entraînement, prédiction
tests/             Tests automatiques (pytest)
resultats/         Chiffres et graphiques de performance
```

## Installation et utilisation

Prérequis : Python 3.10 ou plus.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python 1_telecharger.py            # quelques minutes : plusieurs centaines de Mo
python 2_preparer.py               # lit des millions de trajets
python 3_entrainer.py              # entraîne et évalue (quelques minutes)

streamlit run app.py               # application web
uvicorn api:app --reload           # API : ouvrir http://127.0.0.1:8000/docs
python -m pytest                   # tests
```

Exemple d'appel à l'API :
```
GET /prediction?station=<nom de la station>&jour=2026-07-15&temperature=24&precipitation=0
```

## Limites et pistes d'amélioration

- Le modèle prédit les **départs**, pas les arrivées ni le nombre de vélos disponibles.
- Il ne connaît pas les événements spéciaux (festivals, grèves, travaux), qui peuvent faire exploser la demande.
- Pistes : ajouter les arrivées, les événements, toutes les stations, et redéployer l'API en ligne.

## Sources des données

- Trajets : **BIXI Montréal**, [Données ouvertes – Historique des déplacements](https://bixi.com/fr/donnees-ouvertes/), licence CC-BY 4.0.
- Météo : [Open-Meteo](https://open-meteo.com/), licence CC-BY 4.0.

## Technologies

Python · pandas · scikit-learn · matplotlib · Streamlit · FastAPI · pytest

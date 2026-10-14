"""Télécharger un gros fichier en affichant la progression."""
import requests

ENTETES = {"User-Agent": "Mozilla/5.0 (projet etudiant bixi-prediction)"}


def telecharger(url, destination):
    if destination.exists():
        print(f"  Déjà téléchargé : {destination.name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporaire = destination.with_name(destination.name + ".part")

    with requests.get(url, stream=True, timeout=60, headers=ENTETES) as reponse:
        reponse.raise_for_status()
        total = int(reponse.headers.get("content-length", 0))
        recu, dernier = 0, -1
        with open(temporaire, "wb") as fichier:
            for bloc in reponse.iter_content(chunk_size=1 << 20):  # 1 Mo à la fois
                fichier.write(bloc)
                recu += len(bloc)
                if total:
                    pourcent = recu * 100 // total
                    if pourcent // 10 != dernier:
                        dernier = pourcent // 10
                        print(f"  {destination.name} : {pourcent} % ({recu / 1e6:.0f} Mo)")

    # On renomme seulement à la fin : un téléchargement coupé ne laisse pas un fichier abîmé.
    temporaire.rename(destination)
    return destination

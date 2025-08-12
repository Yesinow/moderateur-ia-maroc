"""
Module principal pour l'application de modération de contenu destinée au contexte
marocain. Cette application propose une API REST ainsi qu'une interface en ligne
de commande pour analyser du texte et des images afin de détecter des
contenus offensants ou illégaux (par exemple nudité, incitation à la haine,
harcèlement, etc.). Elle intègre également une base de connaissances sur
plusieurs codes juridiques marocains afin d'appuyer la prise de décision.

Pour exécuter le serveur :

    uvicorn moderator_maroc_app.app:api --reload

Pour utiliser en ligne de commande :

    python -m moderator_maroc_app.app --text "Votre texte" --image /chemin/vers/fichier.jpg

Cette application nécessite les librairies FastAPI, Pillow et, si
disponible, pytesseract. Pydantic v2 est pris en charge.
"""

import argparse
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# Chargement optionnel de l'OCR si disponible
try:
    import pytesseract  # type: ignore[import]
except Exception:
    pytesseract = None

# Chargement optionnel des modèles d'IA avancés
try:
    # Le modèle multilingual de détection de toxicité/toxicité (transformers). Ce modèle
    # renvoie des étiquettes telles que `toxic`, `obscene`, `insult`, `identity_hate`, etc.
    # La bibliothèque transformers et le modèle doivent être installés séparément.
    from transformers import pipeline  # type: ignore[import]
    TEXT_AI_MODEL = pipeline(
        "text-classification",
        model="martin-ha/toxic-xlm-roberta-base",
        top_k=None  # retourne toutes les étiquettes avec leurs scores
    )
except Exception:
    TEXT_AI_MODEL = None

try:
    # Chargement d'un classifieur de nudité plus avancé. NudeNet fournit un classifieur
    # entraîné pour détecter du contenu NSFW/nudité. Si la librairie n'est pas
    # installée ou qu'une erreur survient, nous utilisons l'heuristique simple.
    from nudenet import NudeClassifier  # type: ignore[import]
    NUDE_CLASSIFIER = NudeClassifier()
except Exception:
    NUDE_CLASSIFIER = None


# -----------------------------------------------------------------------------
# Chargement des règles et des données
# -----------------------------------------------------------------------------

# Politique de modération : catégories et listes de mots clés ou expressions
POLICY_CONFIG: Dict[str, Dict[str, List[str]]] = {
    "nudité": {
        "mots_cles": [
            "nudité", "nu", "exhibition",
            # Ajoutez ici des mots ou expressions jugés explicites
        ],
        "regex": [
            r"\bpose(?:r)?\s*nue?\b",  # exemple: "poser nue"
        ],
    },
    "violence": {
        "mots_cles": [
            "meurtre", "assassiner", "tuer", "agression", "violence", "torture", "terrorisme"
        ],
        "regex": [
            r"\binciter\s+à\s+la\s+violence\b"
        ],
    },
    "harcèlement": {
        "mots_cles": [
            "harceler", "insulte", "injure", "menace", "diffamation"
        ],
        "regex": [
            r"\bharcèlement\s+en\s+ligne\b"
        ],
    },
    "discrimination": {
        "mots_cles": [
            "racisme", "raciste", "sexisme", "sexiste", "homophobie", "xénophobie", "haine raciale"
        ],
        "regex": []
    },
    "drogue": {
        "mots_cles": [
            "drogue", "cannabis", "coke", "ecstasy", "héroïne"
        ],
        "regex": []
    },
    # Ajoutez d'autres catégories au besoin
}

# Catégories considérées comme bloquantes. Si un contenu déclenche l'une de ces
# catégories, il sera marqué comme non valide par défaut.
BLOCKING_CATEGORIES: List[str] = [
    "nudité",
    "violence",
    "harcèlement",
    "discrimination",
    "drogue",
    # Les mots grossiers et injurieux sont également considérés comme bloquants
    "langage grossier",
]

# Chargement de la base de données des lois marocaines
DATA_DIR = Path(__file__).resolve().parent
LAW_DB_PATH = DATA_DIR / "moroccan_laws.json"
BAD_WORDS_PATH = DATA_DIR / "bad_words.json"
try:
    with LAW_DB_PATH.open("r", encoding="utf-8") as f:
        MOROCCAN_LAWS: List[Dict[str, object]] = json.load(f)
except Exception as exc:
    raise RuntimeError(f"Impossible de charger {LAW_DB_PATH}: {exc}")

# Chargement du dictionnaire de mots grossiers/mots injurieux (multilingue)
try:
    with BAD_WORDS_PATH.open("r", encoding="utf-8") as f:
        BAD_WORDS: Dict[str, List[str]] = json.load(f)
except Exception:
    # Si le fichier n'est pas présent ou ne peut pas être lu, utiliser un
    # dictionnaire vide pour ne pas interrompre l'application.
    BAD_WORDS = {}


# -----------------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------------

def extract_image_text(image: Image.Image) -> str:
    """
    Extrait le texte d'une image à l'aide de l'OCR si pytesseract est disponible.
    Si l'OCR n'est pas installé ou qu'une erreur survient, retourne une chaîne vide.
    """
    if pytesseract is None:
        return ""
    try:
        return pytesseract.image_to_string(image, lang="fra+ara")
    except Exception:
        return ""


def detect_nudity_simple(image: Image.Image) -> float:
    """
    Détecte grossièrement la proportion de pixels de type "peau" dans une image
    en utilisant l'espace couleur YCrCb. Cette heuristique rapide indique la
    probabilité de nudité : un score proche de 1 signifie une forte présence de
    peau, un score proche de 0 signifie très peu de peau. Elle sert de
    solution de repli lorsque les modèles plus avancés ne sont pas disponibles.
    """
    # Conversion en YCrCb
    im = image.convert("YCbCr")
    width, height = im.size
    pixels = im.getdata()
    skin_pixels = 0
    total_pixels = width * height
    for ycbcr in pixels:
        y, cb, cr = ycbcr
        # Détection approximative de la couleur de peau. Ces plages proviennent
        # d'études empiriques : elles englobent différentes teintes de peau.
        if 80 <= cb <= 135 and 135 <= cr <= 180 and y > 80:
            skin_pixels += 1
    return skin_pixels / total_pixels if total_pixels else 0.0


def detect_nudity(image: Image.Image) -> float:
    """
    Retourne un score de nudité en utilisant un classifieur avancé lorsqu'il
    est disponible. Lorsque NUDE_CLASSIFIER est chargé avec succès, la
    méthode classify() de NudeNet est utilisée pour estimer la probabilité
    qu'une image contienne du contenu explicite. Si aucune bibliothèque
    avancée n'est disponible, la fonction se rabat sur detect_nudity_simple.

    Le score retourné est normalisé dans l'intervalle [0, 1], où 1 indique
    une probabilité élevée de nudité.
    """
    # Utilisation du classifieur avancé si disponible
    if NUDE_CLASSIFIER is not None:
        try:
            # NudeClassifier.classify retourne un dictionnaire indexé par le
            # chemin de l'image ou l'objet avec des scores pour les clés
            # "safe" et "unsafe". Nous convertissons l'image en mémoire
            # temporaire sous forme de bytes puis la passons au classifieur.
            with io.BytesIO() as tmp_buf:
                image.save(tmp_buf, format="JPEG")
                tmp_bytes = tmp_buf.getvalue()
            result = NUDE_CLASSIFIER.classify(tmp_bytes)
            # Le résultat est un dict { 'data': { 'safe': score_safe, 'unsafe': score_unsafe } }
            # selon les versions, il peut aussi être { <id>: { 'safe': ..., 'unsafe': ... } }
            if isinstance(result, dict):
                # Extraire la première entrée disponible
                data = None
                if 'data' in result:
                    data = result['data']
                else:
                    # Obtenir la première valeur de résultat.values()
                    for v in result.values():
                        data = v
                        break
                if data and 'unsafe' in data:
                    unsafe_score = data['unsafe']
                    # Le classifieur renvoie déjà un score entre 0 et 1
                    return float(unsafe_score)
        except Exception:
            # En cas d'erreur, on redescend sur l'heuristique simple
            pass
    # Fallback sur l'heuristique basique
    return detect_nudity_simple(image)


def analyze_text(text: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Analyse un texte et retourne une liste de catégories déclenchées ainsi qu'un
    dictionnaire de preuves (mots clés ou motifs regex trouvés). Les catégories
    proviennent de POLICY_CONFIG et du dictionnaire BAD_WORDS. Un modèle
    d'IA externe peut également enrichir la classification lorsque disponible.
    """
    triggered_categories: List[str] = []
    evidence: Dict[str, List[str]] = {}
    lower_text = text.lower()
    # 1. Vérification basée sur les règles statiques (POLICY_CONFIG)
    for category, rules in POLICY_CONFIG.items():
        matches: List[str] = []
        # Vérifier les mots clés simples
        for kw in rules.get("mots_cles", []):
            if kw.lower() in lower_text:
                matches.append(kw)
        # Vérifier les expressions régulières
        for pattern in rules.get("regex", []):
            try:
                for match in re.finditer(pattern, lower_text, re.IGNORECASE | re.UNICODE):
                    matches.append(match.group(0))
            except re.error:
                continue
        if matches:
            triggered_categories.append(category)
            evidence[category] = matches
    # 2. Détection de mots injurieux dans toutes les langues disponibles
    bad_matches: List[str] = []
    for lang, words in BAD_WORDS.items():
        for w in words:
            w_low = w.lower()
            if w_low in lower_text:
                bad_matches.append(w)
    if bad_matches:
        triggered_categories.append("langage grossier")
        evidence["langage grossier"] = bad_matches
    # 3. Utilisation facultative d'un classifieur IA pour détecter la toxicité
    if TEXT_AI_MODEL is not None:
        try:
            ai_results = TEXT_AI_MODEL(text)
            # Le pipeline renvoie une liste de dictionnaires {label, score}
            for res in ai_results:
                label = res.get("label", "").lower()
                score = float(res.get("score", 0.0))
                # Seulement considérer des scores élevés
                if score < 0.5:
                    continue
                if label in {"toxic", "severe_toxic", "insult", "obscene", "threat"}:
                    if "harcèlement" not in triggered_categories:
                        triggered_categories.append("harcèlement")
                        evidence.setdefault("harcèlement", []).append(f"{label}:{score:.2f}")
                    else:
                        evidence["harcèlement"].append(f"{label}:{score:.2f}")
                elif label == "identity_hate":
                    if "discrimination" not in triggered_categories:
                        triggered_categories.append("discrimination")
                        evidence.setdefault("discrimination", []).append(f"{label}:{score:.2f}")
                    else:
                        evidence["discrimination"].append(f"{label}:{score:.2f}")
        except Exception:
            # En cas d'erreur dans le modèle, continuer avec les règles statiques
            pass
    return triggered_categories, evidence


def find_law_references(text: str) -> List[str]:
    """
    Cherche des références aux lois marocaines dans un texte. Retourne la liste
    des noms des lois (définies dans MOROCCAN_LAWS) trouvées.
    """
    references: List[str] = []
    lower_text = text.lower()
    for entry in MOROCCAN_LAWS:
        for synonym in entry.get("synonyms", []):
            if synonym.lower() in lower_text:
                references.append(entry["name"])  # type: ignore[index]
                break
    return references


# -----------------------------------------------------------------------------
# API et modèles Pydantic
# -----------------------------------------------------------------------------

class AnalysisInput(BaseModel):
    title: Optional[str] = None
    text: Optional[str] = None
    image_url: Optional[str] = None


class AnalysisResult(BaseModel):
    valide: bool
    categories: List[str]
    motifs: Dict[str, List[str]]
    lois_detectees: List[str]
    score_nudite: Optional[float] = None
    remarque: Optional[str] = None


api = FastAPI(title="Modérateur Marocain", version="1.0.0")


@api.get("/health")
def health_check() -> Dict[str, str]:
    """
    Endpoint simple pour vérifier que l'API fonctionne.
    """
    return {"status": "ok"}


@api.get("/laws")
def list_laws() -> List[Dict[str, object]]:
    """
    Renvoie la base de données des lois marocaines incluse dans l'application.
    """
    return MOROCCAN_LAWS


@api.post("/analyze")
async def analyze_payload(payload: AnalysisInput) -> AnalysisResult:
    """
    Analyse une requête JSON contenant éventuellement un titre, un texte et une
    URL d'image. Les images distantes ne sont pas téléchargées dans cette
    implémentation : seules les analyses de texte sont effectuées si aucune
    image n'est fournie via le second endpoint multipart. Si une URL d'image
    est fournie, le client doit d'abord la télécharger et l'envoyer via
    `/analyze-multipart`.
    """
    if not payload.title and not payload.text:
        raise HTTPException(status_code=400, detail="Veuillez fournir au moins un titre ou un texte à analyser.")

    combined_text = " ".join(filter(None, [payload.title, payload.text]))

    categories, evidence = analyze_text(combined_text)
    laws = find_law_references(combined_text)

    # Score de nudité non disponible sans image
    score_nudite = None

    # Déterminer si valide : un contenu est non valide si une catégorie bloquante est déclenchée
    is_valid = not any(cat in BLOCKING_CATEGORIES for cat in categories)

    remarque = None
    if not is_valid:
        motifs_blockants = [cat for cat in categories if cat in BLOCKING_CATEGORIES]
        remarque = f"Contenu non valide en raison des catégories suivantes : {', '.join(motifs_blockants)}."

    return AnalysisResult(
        valide=is_valid,
        categories=categories,
        motifs=evidence,
        lois_detectees=laws,
        score_nudite=score_nudite,
        remarque=remarque,
    )


# Le support des formulaires multipart (pour l'envoi d'images via HTTP) est
# volontairement omis afin d'éviter d'imposer la dépendance "python-multipart".
# Les images doivent être analysées via l'interface en ligne de commande ou
# encodées en base64 et envoyées dans le corps JSON de l'endpoint /analyze
# (cette option n'est pas implémentée ici).


# -----------------------------------------------------------------------------
# Exécution en ligne de commande
# -----------------------------------------------------------------------------

def cli_main() -> None:
    """
    Entrée principale pour l'utilisation en tant que script. Permet d'analyser
    des textes et des images depuis la ligne de commande sans lancer le
    serveur FastAPI.
    """
    parser = argparse.ArgumentParser(description="Analyseur de contenus marocain")
    parser.add_argument(
        "--title", dest="title", help="Titre à analyser", required=False
    )
    parser.add_argument(
        "--text", dest="text", help="Contenu textuel à analyser", required=False
    )
    parser.add_argument(
        "--image", dest="image_path", help="Chemin vers un fichier image", required=False
    )
    args = parser.parse_args()
    parts: List[str] = []
    if args.title:
        parts.append(args.title)
    if args.text:
        parts.append(args.text)
    image_score: Optional[float] = None
    if args.image_path:
        try:
            with open(args.image_path, "rb") as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            # Utiliser la fonction de détection la plus adaptée
            image_score = detect_nudity(img)
            ocr_txt = extract_image_text(img)
            if ocr_txt:
                parts.append(ocr_txt)
        except Exception as exc:
            print(f"Erreur lors de l'ouverture de l'image: {exc}")
    # Si aucun texte n'est fourni mais qu'une image a été traitée, nous
    # poursuivons l'analyse avec un texte vide afin de retourner un résultat
    if not parts and args.image_path:
        parts.append("")
    if not parts:
        print("Aucun contenu fourni.")
        return
    combined = " ".join(parts)
    cats, ev = analyze_text(combined)
    laws = find_law_references(combined)
    # Si un score de nudité important est détecté (> 0.3), on considère la catégorie "nudité" déclenchée
    if image_score is not None and image_score > 0.3:
        cats.append("nudité")
        ev.setdefault("nudité", []).append(f"score={image_score:.2f}")
    is_valid = not any(c in BLOCKING_CATEGORIES for c in cats)
    result = {
        "valide": is_valid,
        "categories": cats,
        "motifs": ev,
        "lois_detectees": laws,
        "score_nudite": image_score,
    }
    # Sortie JSON
    try:
        import json as _json  # éviter l'ombre d'import
        print(_json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print(result)


if __name__ == "__main__":
    cli_main()
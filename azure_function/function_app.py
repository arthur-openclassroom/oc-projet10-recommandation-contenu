"""Azure Function de recommandation My Content (architecture B).

La fonction lit l'instantané du modèle (snapshot.npz) directement dans
Blob Storage, sans API intermédiaire. L'instantané est gardé en mémoire et
n'est retéléchargé que s'il a changé (vérification toutes les 60 s).

Paramètres : user_id (obligatoire) et recent (facultatif : articles lus
depuis l'instantané). La réponse donne les 5 articles et, pour chacun, le
signal qui l'a fait retenir.
"""

import hashlib
import json
import logging
import re
import string
import threading
import time

import azure.functions as func
import azurefunctions.extensions.bindings.blob as blob

import recommender

SNAPSHOT_PATH = "artifacts/snapshot.npz"
STORAGE_CONNECTION = "AzureWebJobsStorage"
REVALIDATION_S = 60.0
TOP_K = 5
MAX_RECENT = 20
MAX_ARTICLE_ID = 2 ** 31 - 1
RECENT_ERROR = (f"paramètre recent invalide : au plus {MAX_RECENT} "
                f"identifiants d'articles, entiers de 0 à {MAX_ARTICLE_ID}, "
                f"séparés par des virgules")
USER_ID_PATTERN = re.compile(r"\s*-?\d{1,19}\s*", re.ASCII)
RECENT_ITEM_PATTERN = re.compile(r"\s*\d{1,10}\s*", re.ASCII)

logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


class InvalidSnapshotError(RuntimeError):
    """Blob d'instantané inchangé depuis l'échec de son chargement."""

    def __init__(self, etag):
        super().__init__(f"instantané invalide (ETag {etag}) : en attente "
                         f"d'une republication")


class SnapshotCache:
    """Instantané du modèle en mémoire, indexé par l'ETag du blob source."""

    def __init__(self, clock=time.monotonic, revalidation_s=REVALIDATION_S):
        self._clock = clock
        self._revalidation_s = revalidation_s
        self._lock = threading.Lock()
        self._etag = None
        self._bad_etag = None
        self._snapshot = None
        self._checked_at = 0.0

    def get(self, client):
        """Renvoie l'instantané, rechargé seulement si le blob a changé."""
        with self._lock:
            now = self._clock()
            if now - self._checked_at < self._revalidation_s:
                if self._snapshot is not None:
                    return self._snapshot
                if self._bad_etag is not None:
                    raise InvalidSnapshotError(self._bad_etag)
            try:
                etag = client.get_blob_properties().etag
                self._checked_at = now
                if etag == self._bad_etag:
                    if self._snapshot is None:
                        raise InvalidSnapshotError(etag)
                elif etag != self._etag:
                    self._bad_etag = None
                    data = client.download_blob().readall()
                    try:
                        self._snapshot = recommender.load_snapshot(data)
                    except Exception:
                        self._bad_etag = etag
                        raise
                    self._etag = etag
                    logger.info("Instantané chargé (ETag %s, %d octets).",
                                etag, len(data))
            except Exception:
                if self._snapshot is None:
                    raise
                self._checked_at = now
                logger.exception("Revalidation de l'instantané impossible : "
                                 "version précédente conservée.")
            return self._snapshot


_cache = SnapshotCache()


def _json_response(body, status_code):
    return func.HttpResponse(
        json.dumps(body, ensure_ascii=False),
        status_code=status_code,
        mimetype="application/json",
    )


def _pseudonym(user_id):
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:10]


def _parse_user_id(value):
    """Identifiant utilisateur (chiffres ASCII) ; None s'il est invalide."""
    if value is None or not USER_ID_PATTERN.fullmatch(value):
        return None
    return int(value)


def _parse_recent(value):
    """Articles du paramètre recent ; None si le paramètre est invalide."""
    if value is None or not value.strip(string.whitespace):
        return []
    parts = value.split(",")
    if len(parts) > MAX_RECENT or not all(RECENT_ITEM_PATTERN.fullmatch(part)
                                          for part in parts):
        return None
    items = [int(part) for part in parts]
    return items if max(items) <= MAX_ARTICLE_ID else None


def _unavailable_response():
    logger.exception("Instantané du modèle indisponible.")
    return _json_response({"error": "instantané du modèle indisponible"}, 503)


def _active_signals(info):
    """Noms des signaux ayant contribué au classement."""
    return (["tendance"] + (["co-visitation"] if info["covisitation"] else [])
            + (["contenu"] if info["cb"] else []))


def _details(recommendations, info):
    """Score, contributions arrondies et signal principal de chaque article."""
    parts = info["contributions"]
    return [
        {
            "article_id": int(article),
            "score": round(info["scores"][i], 4),
            "tendance": round(parts["tendance"][i], 4),
            "covisitation": round(parts["covisitation"][i], 4),
            "contenu": round(parts["contenu"][i], 4),
            "signal_principal": info["signal_principal"][i],
        }
        for i, article in enumerate(recommendations)
    ]


@app.route(route="recommend", methods=["GET"],
           auth_level=func.AuthLevel.FUNCTION)
@app.blob_input(arg_name="client", path=SNAPSHOT_PATH,
                connection=STORAGE_CONNECTION)
def recommend_articles(req: func.HttpRequest,
                       client: blob.BlobClient) -> func.HttpResponse:
    """Renvoie les 5 articles recommandés pour user_id (recent facultatif)."""
    start = time.perf_counter()
    user_id = _parse_user_id(req.params.get("user_id"))
    if user_id is None:
        return _json_response({"error": "paramètre user_id entier requis"},
                              400)
    recent = _parse_recent(req.params.get("recent"))
    if recent is None:
        return _json_response({"error": RECENT_ERROR}, 400)

    try:
        snapshot = _cache.get(client)
    except Exception:
        return _unavailable_response()

    recommendations, source, info = recommender.recommend(
        snapshot, user_id, k=TOP_K, recent_items=recent, return_info=True)
    meta = recommender.snapshot_info(snapshot)
    logger.info("Recommandation servie : utilisateur %s, %d lecture(s) "
                "récente(s), source %s, signaux %s, %.1f ms.",
                _pseudonym(user_id), len(recent), source,
                ", ".join(_active_signals(info)),
                (time.perf_counter() - start) * 1000)
    return _json_response(
        {
            "user_id": user_id,
            "recommendations": [int(a) for a in recommendations],
            "source": source,
            "snapshot": {
                "t_ref": meta.get("t_ref"),
                "version": meta.get("version"),
            },
            "signaux": {
                "covisitation": bool(info["covisitation"]),
                "contenu": bool(info["cb"]),
                "nb_candidats": int(info["n_candidats"]),
            },
            "details": _details(recommendations, info),
        },
        200,
    )


@app.route(route="health", methods=["GET"],
           auth_level=func.AuthLevel.FUNCTION)
@app.blob_input(arg_name="client", path=SNAPSHOT_PATH,
                connection=STORAGE_CONNECTION)
def health(req: func.HttpRequest,
           client: blob.BlobClient) -> func.HttpResponse:
    """Renvoie les métadonnées de l'instantané actuellement servi."""
    try:
        snapshot = _cache.get(client)
    except Exception:
        return _unavailable_response()
    return _json_response(recommender.snapshot_info(snapshot), 200)

"""Modèle hybride My Content : tendance, co-visitation et contenu.

score = tendance + 0.7 * co-visitation + 0.008 * contenu
La co-visitation ne compte que si le dernier clic date de 30 min au plus ;
un nouvel utilisateur reçoit la tendance seule.

Utilisé par l'Azure Function, scripts/prepare_artifacts.py et le notebook 04.
"""

import io
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np

MS_MIN = 60_000
MS_H = 3_600_000
K = 5
MODEL_VERSION = "hybride-2.1"
SIGNALS = ("tendance", "co-visitation", "contenu")

CONFIG = dict(
    trend_window_ms=2 * MS_H,
    trend_tau_ms=5 * MS_MIN,
    n_trend=100,
    cv_window_ms=24 * MS_H,
    cv_tau_ms=2 * MS_H,
    cv_top_m=20,
    w_cv=0.7,
    gate_ms=30 * MS_MIN,
    fresh_ms=12 * MS_H,
    w_cb=0.008,
)
REQUIRED_KEYS = {
    "version", "config", "t_ref", "trend_ids", "trend_score",
    "cv_src", "cv_dst", "cv_w", "user_ids", "user_last_ts",
    "hist_users", "hist_items", "art_ids", "art_created", "art_emb",
}


# Construction de l'instantané (en local)

def trend_signal(ts, articles, t_ref, cfg):
    """Tendance : part des clics des 2 dernières heures avant t_ref.

    Un clic compte d'autant plus qu'il est récent. Les articles sont
    classés du plus au moins tendance.
    """
    recent = (ts >= t_ref - cfg["trend_window_ms"]) & (ts < t_ref)
    if not recent.any():
        return np.empty(0, np.int64), np.empty(0)
    weights = np.exp(-(t_ref - ts[recent]) / cfg["trend_tau_ms"])
    # np.unique numérote les articles, bincount somme les poids par numéro
    ids, num = np.unique(articles[recent], return_inverse=True)
    score = np.bincount(num, weights=weights)
    share = score / score.sum()
    order = np.argsort(-share, kind="stable")  # à égalité, plus petit id
    return ids[order].astype(np.int64), share[order]


def session_pairs(user, session, article, ts):
    """Paires de clics consécutifs d'une même session.

    Les clics sont triés par utilisateur puis par date. Renvoie (article lu,
    article lu juste après, date du second clic), triés par date.
    """
    same = ((user[1:] == user[:-1]) & (session[1:] == session[:-1])
            & (article[1:] != article[:-1]))
    src, dst, pair_ts = article[:-1][same], article[1:][same], ts[1:][same]
    order = np.argsort(pair_ts, kind="stable")
    return src[order], dst[order], pair_ts[order]


def covisitation(src, dst, pair_ts, t_ref, cfg):
    """Co-visitation : pour chaque article, les articles lus juste après.

    Une ligne par paire (source, voisin), avec la probabilité que le voisin
    soit lu après la source sur les 24 dernières heures. On garde les 20
    meilleurs voisins par source.
    """
    import pandas as pd  # seulement en local, pas sur Azure

    recent = (pair_ts >= t_ref - cfg["cv_window_ms"]) & (pair_ts < t_ref)
    pairs = pd.DataFrame({
        "src": src[recent], "dst": dst[recent],
        "w": np.exp(-(t_ref - pair_ts[recent]) / cfg["cv_tau_ms"]),
    })
    pairs = pairs.groupby(["src", "dst"], as_index=False)["w"].sum()
    pairs["w"] /= pairs.groupby("src")["w"].transform("sum")
    pairs = pairs.sort_values(["src", "w", "dst"],
                              ascending=[True, False, True])
    pairs = pairs.groupby("src").head(cfg["cv_top_m"])
    return (pairs["src"].to_numpy(np.int64), pairs["dst"].to_numpy(np.int64),
            pairs["w"].to_numpy(np.float64))


def effective_end(ts, min_per_hour=1000):
    """Fin des logs : fin de la dernière heure à min_per_hour clics."""
    hours, counts = np.unique(np.asarray(ts) // MS_H, return_counts=True)
    return int((hours[counts >= min_per_hour].max() + 1) * MS_H)


def build_snapshot(clicks, art_ids, art_created, art_emb, t_ref, cfg=CONFIG):
    """Construit l'instantané du modèle à la date t_ref (en ms)."""
    user = np.asarray(clicks["user_id"], dtype=np.int64)
    session = np.asarray(clicks["session_id"], dtype=np.int64)
    article = np.asarray(clicks["click_article_id"], dtype=np.int64)
    ts = np.asarray(clicks["click_timestamp"], dtype=np.int64)
    before = ts < t_ref
    user, session, article, ts = (user[before], session[before],
                                  article[before], ts[before])

    trend_ids, trend_score = trend_signal(ts, article, t_ref, cfg)

    o = np.lexsort((ts, user))  # par utilisateur, puis par date
    user, session, article, ts = user[o], session[o], article[o], ts[o]
    cv_src, cv_dst, cv_w = covisitation(
        *session_pairs(user, session, article, ts), t_ref, cfg)
    import pandas as pd  # seulement en local, pas sur Azure
    last = pd.Series(ts).groupby(user).max()
    user_ids, user_last_ts = last.index.to_numpy(), last.to_numpy()

    o = np.argsort(np.asarray(art_ids), kind="stable")
    art_ids = np.asarray(art_ids, dtype=np.int64)[o]
    art_created = np.asarray(art_created, dtype=np.int64)[o]
    art_emb = np.asarray(art_emb, dtype=np.float32)[o]
    fresh = (art_created >= t_ref - cfg["fresh_ms"]) & (art_created <= t_ref)
    useful = np.isin(art_ids, article) | fresh  # articles lus ou frais
    emb = art_emb[useful]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return dict(
        version=np.array(MODEL_VERSION),
        config=np.array(json.dumps(dict(cfg))),
        t_ref=np.int64(t_ref),
        trend_ids=trend_ids, trend_score=trend_score,
        cv_src=cv_src, cv_dst=cv_dst, cv_w=cv_w,
        user_ids=user_ids.astype(np.int32), user_last_ts=user_last_ts,
        hist_users=user.astype(np.int32), hist_items=article.astype(np.int32),
        art_ids=art_ids[useful].astype(np.int32),
        art_created=art_created[useful],
        art_emb=(emb / norms).astype(np.float16))


def save_snapshot(snap, path):
    """Écrit l'instantané dans un fichier npz compressé."""
    np.savez_compressed(path, **snap)


# Lecture de l'instantané et recommandation (sur Azure)

def load_snapshot(data):
    """Charge un instantané (chemin ou contenu binaire).

    Lève ValueError s'il manque une clé ou un réglage : un instantané
    incomplet n'est jamais servi.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        data = io.BytesIO(data)
    else:
        data = os.fspath(data)
    with np.load(data, allow_pickle=False) as z:
        missing = REQUIRED_KEYS.difference(z.files)
        if missing:
            raise ValueError("instantané incomplet : clé(s) manquante(s) "
                             + ", ".join(sorted(missing)))
        snap = {key: z[key] for key in z.files}
    cfg = json.loads(str(snap["config"]))
    if set(CONFIG) - set(cfg):
        raise ValueError("instantané incomplet : configuration incomplète")
    return snap


def user_history(snap, user_id):
    """Articles lus par l'utilisateur et date de son dernier clic.

    Pour un utilisateur inconnu : liste vide et None.
    """
    if not 0 <= user_id < 2 ** 31:
        return np.empty(0, np.int64), None
    uid = np.int32(user_id)  # même type que hist_users : pas de copie
    lo = np.searchsorted(snap["hist_users"], uid, side="left")
    hi = np.searchsorted(snap["hist_users"], uid, side="right")
    items = snap["hist_items"][lo:hi].astype(np.int64)
    if len(items) == 0:
        return items, None
    p = np.searchsorted(snap["user_ids"], uid)
    return items, int(snap["user_last_ts"][p])


def covis_neighbors(snap, article):
    """Voisins de co-visitation d'un article : {voisin: probabilité}."""
    lo = np.searchsorted(snap["cv_src"], article, side="left")
    hi = np.searchsorted(snap["cv_src"], article, side="right")
    return dict(zip(snap["cv_dst"][lo:hi].tolist(),
                    snap["cv_w"][lo:hi].tolist()))


def user_profile(emb, rows):
    """Profil de contenu : somme normalisée des articles distincts lus."""
    rows = list(dict.fromkeys(rows))  # sans doublon, dans l'ordre de lecture
    profile = emb[rows].astype(np.float32).sum(axis=0, dtype=np.float64)
    norm = np.linalg.norm(profile)
    return profile / norm if norm > 0 else profile


def z_score(x):
    """Valeurs centrées-réduites (0 si elles sont toutes égales)."""
    if len(x) < 2 or x.std() == 0:
        return np.zeros(len(x))
    return (x - x.mean()) / x.std()


def _find(sorted_ids, ids):
    """Position de chaque id dans sorted_ids, et masque des ids trouvés."""
    pos = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
    return pos, sorted_ids[pos] == ids


def recommend(snapshot, user_id, k=K, recent_items=None, now_ms=None,
              return_info=False):
    """Renvoie les k articles recommandés et la source.

    La source vaut "hybride" si l'utilisateur a un historique, "tendance"
    sinon. recent_items : articles lus depuis l'instantané. Avec
    return_info=True, renvoie aussi le détail des scores.
    """
    cfg = json.loads(str(snapshot["config"]))
    t = int(snapshot["t_ref"]) if now_ms is None else int(now_ms)
    items, last_click = user_history(snapshot, int(user_id))
    if recent_items is not None and len(recent_items):
        recent = np.asarray(recent_items, dtype=np.int64).ravel()
        items = np.concatenate([items, recent])
        last_click = t
    has_hist = len(items) > 0
    art_ids = snapshot["art_ids"].astype(np.int64)
    created = snapshot["art_created"]

    # 1. Candidats : tendance, articles récents et voisins de co-visitation
    trend_ids, trend_score = snapshot["trend_ids"], snapshot["trend_score"]
    fresh = art_ids[(created >= t - cfg["fresh_ms"]) & (created <= t)]
    neighbors = {}
    if has_hist and cfg["w_cv"] > 0 and t - last_click <= cfg["gate_ms"]:
        neighbors = covis_neighbors(snapshot, int(items[-1]))
    covis_on = len(neighbors) > 0
    cand = np.unique(np.concatenate([
        trend_ids[:cfg["n_trend"]], fresh,
        np.array(list(neighbors), dtype=np.int64)]))

    # Exclus : articles inconnus, pas encore publiés ou déjà lus
    pos, known = _find(art_ids, cand)
    keep = known & (created[pos] <= t) & ~np.isin(cand, items)
    cand, pos = cand[keep], pos[keep]

    # 2. Les trois notes de chaque candidat
    trend = dict(zip(trend_ids.tolist(), trend_score.tolist()))
    trend_part = np.array([trend.get(a, 0.0) for a in cand.tolist()])
    covis_part = np.array([cfg["w_cv"] * neighbors.get(a, 0.0)
                           for a in cand.tolist()])
    content_part = np.zeros(len(cand))
    content_on = has_hist and cfg["w_cb"] > 0 and len(cand) > 0
    if content_on:
        h_pos, h_known = _find(art_ids, items)
        emb = snapshot["art_emb"]
        profile = user_profile(emb, h_pos[h_known].tolist())
        cos = emb[pos].astype(np.float32) @ profile.astype(np.float32)
        content_part = cfg["w_cb"] * z_score(cos.astype(np.float64))
    score = trend_part + covis_part + content_part

    # 3. Les k meilleurs ; à égalité, le plus récent puis le plus petit id
    best = sorted(range(len(cand)),
                  key=lambda i: (-score[i], -created[pos[i]], cand[i]))[:k]
    ids = [int(cand[i]) for i in best]
    source = "hybride" if has_hist else "tendance"
    if not return_info:
        return ids, source
    parts = {"tendance": trend_part[best], "covisitation": covis_part[best],
             "contenu": content_part[best]}
    signal = [SIGNALS[int(np.argmax(col))] for col in
              zip(trend_part[best], covis_part[best], content_part[best])]
    info = dict(covisitation=covis_on, cb=content_on,
                n_candidats=len(cand), scores=score[best].tolist(),
                contributions={key: value.tolist()
                               for key, value in parts.items()},
                signal_principal=signal)
    return ids, source, info


def snapshot_info(snapshot):
    """Métadonnées de l'instantané (date, version, volumes)."""
    cfg = json.loads(str(snapshot["config"]))
    t_ref = int(snapshot["t_ref"])
    created = snapshot["art_created"]
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return {
        "t_ref": (epoch + timedelta(milliseconds=t_ref)).isoformat(),
        "version": str(snapshot["version"]),
        "nb_utilisateurs": len(snapshot["user_ids"]),
        "nb_articles": len(snapshot["art_ids"]),
        "nb_candidats_tendance": min(cfg["n_trend"],
                                     len(snapshot["trend_ids"])),
        "nb_articles_frais": int(((created >= t_ref - cfg["fresh_ms"])
                                  & (created <= t_ref)).sum()),
    }

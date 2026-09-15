"""Fonctions partagées par les notebooks : chargement des données, split
leave-last-out (protocole v1 des notebooks 02 et 03) et métriques @ 5.
"""

import glob
import os
import pickle

import numpy as np
import pandas as pd

SEED = 12
N_RECO = 5


def load_clicks(data_dir="data"):
    """Charge et concatène l'ensemble des fichiers de clics horaires."""
    files = sorted(glob.glob(os.path.join(data_dir, "clicks", "*.csv")))
    clicks = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return clicks.sort_values("click_timestamp").reset_index(drop=True)


def load_embeddings(data_dir="data"):
    """Charge la matrice d'embeddings des articles (article_id = index de ligne)."""
    with open(os.path.join(data_dir, "articles_embeddings.pickle"), "rb") as f:
        return pickle.load(f)


def load_articles_metadata(data_dir="data"):
    return pd.read_csv(os.path.join(data_dir, "articles_metadata.csv"))


def temporal_split(clicks):
    """Split leave-last-out : le dernier clic de chaque utilisateur sert de test.

    Seuls les utilisateurs ayant au moins 2 clics ont une ligne de test.
    Retourne (train, test), deux DataFrames aux mêmes colonnes que `clicks`.
    """
    clicks = clicks.sort_values(["user_id", "click_timestamp"])
    last = clicks.groupby("user_id").tail(1)
    counts = clicks["user_id"].value_counts()
    eligible = counts[counts >= 2].index
    test = last[last["user_id"].isin(eligible)]
    train = clicks.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def sample_eval_users(test, n=5000, seed=SEED):
    """Échantillonne les utilisateurs sur lesquels évaluer (reproductible)."""
    rng = np.random.default_rng(seed)
    users = test["user_id"].unique()
    if len(users) <= n:
        return users
    return rng.choice(users, size=n, replace=False)


def hit_rate_at_k(recommendations, test_items, k=N_RECO):
    """Part des utilisateurs dont l'article test figure dans le top-k recommandé.

    recommendations : dict user_id -> liste ordonnée d'article_ids.
    test_items : dict user_id -> article_id attendu.
    """
    hits = sum(
        1 for u, recos in recommendations.items() if test_items[u] in recos[:k]
    )
    return hits / len(recommendations)


def mrr_at_k(recommendations, test_items, k=N_RECO):
    """Mean Reciprocal Rank : 1/rang de l'article test s'il est dans le top-k."""
    total = 0.0
    for u, recos in recommendations.items():
        target = test_items[u]
        for rank, art in enumerate(recos[:k], start=1):
            if art == target:
                total += 1.0 / rank
                break
    return total / len(recommendations)

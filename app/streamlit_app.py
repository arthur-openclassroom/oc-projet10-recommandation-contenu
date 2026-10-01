"""Application My Content.

Liste les identifiants des utilisateurs, appelle l'Azure Function de
recommandation pour l'utilisateur choisi et affiche les 5 articles suggérés.
L'adresse et la clé de la fonction sont lues dans le fichier .env
(variables FUNCTION_URL et FUNCTION_KEY).
"""

import os

import pandas as pd
import requests
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(os.path.dirname(APP_DIR), ".env")
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, encoding="utf-8") as env_file:
        for line in env_file:
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key, value)
FUNCTION_URL = os.environ.get("FUNCTION_URL",
                              "http://localhost:7071/api/recommend")
FUNCTION_KEY = os.environ.get("FUNCTION_KEY")


@st.cache_data
def load_user_ids():
    return pd.read_csv(os.path.join(APP_DIR, "utilisateurs.csv"))[
        "user_id"].tolist()


@st.cache_data
def load_articles():
    return pd.read_csv(os.path.join(APP_DIR, "articles.csv")).set_index(
        "article_id")


st.title("My Content : recommandation d'articles")

user_id = st.selectbox("Utilisateur", load_user_ids(), index=None,
                       placeholder="Choisir un identifiant")

if user_id is not None:
    headers = {"x-functions-key": FUNCTION_KEY} if FUNCTION_KEY else {}
    try:
        with st.spinner("Appel de l'Azure Function..."):
            response = requests.get(FUNCTION_URL, params={"user_id": user_id},
                                    headers=headers, timeout=60)
    except requests.RequestException as exc:
        st.error(f"La fonction ne répond pas : {exc}")
        st.stop()
    if response.status_code != 200:
        st.error(f"Erreur {response.status_code} : {response.text[:200]}")
        st.stop()

    ids = response.json()["recommendations"]
    articles = load_articles().reindex(ids)
    published = pd.to_datetime(articles["created_at_ts"], unit="ms")
    st.subheader(f"5 articles pour l'utilisateur {user_id}")
    st.dataframe(pd.DataFrame({
        "Rang": range(1, len(ids) + 1),
        "Article": ids,
        "Catégorie": articles["category_id"].to_numpy(),
        "Nombre de mots": articles["words_count"].to_numpy(),
        "Publié le": published.dt.strftime("%d/%m/%Y %H:%M").to_numpy(),
    }), hide_index=True)

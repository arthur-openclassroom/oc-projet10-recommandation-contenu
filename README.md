# My Content : recommandation d'articles

MVP de recommandation de contenu : une application Streamlit appelle une Azure Function (serverless) qui renvoie 5 articles pour un utilisateur.

Dépôt : https://github.com/arthur-openclassroom/oc-projet10-recommandation-contenu

## Lancer le projet

1. Données : télécharger le jeu Globo.com sur Kaggle (`news-portal-user-interactions-by-globocom`), puis
   `unzip news-portal-user-interactions-by-globocom.zip -d data && unzip data/clicks.zip -d data`
2. Environnement (Python 3.11) : `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`
3. Notebooks `01` à `04` : exploration, modèles testés, choix du modèle hybride.
4. Déploiement sur Azure : `./deploy.sh` (prépare le modèle, déploie la fonction, écrit l'adresse et la clé dans `.env`).
5. Application : `.venv/bin/python -m streamlit run app/streamlit_app.py`

Après la démo : `az group delete --name rg-mycontent-p10 --yes`

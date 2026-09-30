import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "azure_function"))
import recommender  # noqa: E402

PCA_COMPONENTS = 50
SEED = 42
CLICK_COLUMNS = ["user_id", "session_id", "click_article_id",
                 "click_timestamp"]
SNAPSHOT_NAME = "snapshot.npz"
CONNECTION_VARIABLE = "AZURE_STORAGE_CONNECTION_STRING"
DEMO_USERS = (1713, 152, 201, 999999999)
APP_DIR = os.path.join(BASE_DIR, "app")
N_APP_USERS = 2000


def load_clicks(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "clicks", "*.csv")))
    if not files:
        sys.exit(f"Aucun fichier de clics trouvé dans {data_dir}/clicks.")
    clicks = pd.concat((pd.read_csv(f, usecols=CLICK_COLUMNS) for f in files),
                       ignore_index=True)
    clicks = clicks.astype({c: "int64" for c in CLICK_COLUMNS})
    clicks = clicks.sort_values("click_timestamp", kind="stable")
    print(f"Clics : {len(clicks)} lignes ({len(files)} fichiers), "
          f"{clicks['user_id'].nunique()} utilisateurs, "
          f"{clicks['click_article_id'].nunique()} articles cliqués")
    return {c: clicks[c].to_numpy(np.int64) for c in CLICK_COLUMNS}


def load_articles(data_dir):
    meta = pd.read_csv(os.path.join(data_dir, "articles_metadata.csv"))
    print(f"Métadonnées : {len(meta)} articles")
    return (meta["article_id"].to_numpy(np.int64),
            meta["created_at_ts"].to_numpy(np.int64))


def fit_pca(data_dir, out_dir):
    """Réduit les embeddings à 50 dimensions et sauvegarde la projection."""
    with open(os.path.join(data_dir, "articles_embeddings.pickle"), "rb") as f:
        embeddings = pickle.load(f)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    path = os.path.join(out_dir, "pca_50.npz")
    np.savez(path, components_=pca.components_, mean_=pca.mean_,
             explained_variance_ratio_=pca.explained_variance_ratio_)
    print(f"PCA : {embeddings.shape} -> {reduced.shape}, variance expliquée "
          f"{pca.explained_variance_ratio_.sum():.1%} ; projection "
          f"sauvegardée dans {path} ({os.path.getsize(path) / 1e3:.0f} Ko)")
    return reduced


def report(path):
    start = time.perf_counter()
    snap = recommender.load_snapshot(path)
    load_ms = (time.perf_counter() - start) * 1000
    print(f"\nInstantané : {path}, {os.path.getsize(path) / 1e6:.2f} Mo "
          f"(npz compressé), chargé en {load_ms:.0f} ms")
    raw = sorted(((v.nbytes, k) for k, v in snap.items()
                  if not k.startswith("_")), reverse=True)
    print("Tailles brutes : " + ", ".join(
        f"{k} {n / 1e6:.2f} Mo" for n, k in raw[:6]))
    print(json.dumps(recommender.snapshot_info(snap), ensure_ascii=False,
                     indent=1))
    print(f"Sources de co-visitation : {len(set(snap['cv_src']))} articles ; "
          f"5 premiers articles de la tendance : "
          f"{recommender.recommend(snap, -1)[0]}")
    print("\nUtilisateurs de démonstration :")
    for user_id in DEMO_USERS:
        start = time.perf_counter()
        ids, source, info = recommender.recommend(snap, user_id,
                                                  return_info=True)
        latency = (time.perf_counter() - start) * 1000
        history, _ = recommender.user_history(snap, user_id)
        print(f"  {user_id:>9} : {ids} source {source}, historique "
              f"{len(history)} clics, co-visitation "
              f"{'active' if info['covisitation'] else 'inactive'}, "
              f"{latency:.2f} ms")


def upload_snapshot(path, container_name):
    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient.from_connection_string(
        os.environ[CONNECTION_VARIABLE])
    container = service.get_container_client(container_name)
    if not container.exists():
        container.create_container()
        print(f"Container '{container_name}' créé")
    with open(path, "rb") as f:
        container.upload_blob(SNAPSHOT_NAME, f, overwrite=True)
    print(f"Envoyé : {path} -> {container_name}/{SNAPSHOT_NAME}")


def export_app_data(snap, data_dir):
    """Écrit les petits fichiers lus par l'application Streamlit."""
    users = np.sort(snap["user_ids"])[:N_APP_USERS]
    pd.DataFrame({"user_id": users}).to_csv(
        os.path.join(APP_DIR, "utilisateurs.csv"), index=False)
    meta = pd.read_csv(os.path.join(data_dir, "articles_metadata.csv"))
    meta = meta[meta["article_id"].isin(snap["art_ids"])]
    meta[["article_id", "category_id", "words_count", "created_at_ts"]].to_csv(
        os.path.join(APP_DIR, "articles.csv"), index=False)
    print(f"Application : {len(users)} utilisateurs et {len(meta)} articles "
          f"écrits dans app/")


def main():
    parser = argparse.ArgumentParser(
        description="Prépare l'instantané du modèle de recommandation.")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data"),
                        help="dossier des données brutes")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "artifacts"),
                        help="dossier de sortie des artefacts")
    parser.add_argument("--upload", action="store_true",
                        help="envoie snapshot.npz vers Azure Blob Storage")
    parser.add_argument("--container", default="artifacts",
                        help="container Azure Blob Storage de destination")
    args = parser.parse_args()
    if args.upload and not os.environ.get(CONNECTION_VARIABLE):
        parser.error(f"--upload nécessite la variable d'environnement "
                     f"{CONNECTION_VARIABLE} (chaîne de connexion du compte "
                     f"de stockage de la Function App).")

    start = time.time()
    os.makedirs(args.out, exist_ok=True)
    clicks = load_clicks(args.data_dir)
    art_ids, art_created = load_articles(args.data_dir)
    embeddings = fit_pca(args.data_dir, args.out)

    ts = clicks["click_timestamp"]
    t_ref = recommender.effective_end(ts)
    print(f"t_ref = {pd.Timestamp(t_ref, unit='ms')} UTC (fin effective des "
          f"logs) ; {int((ts >= t_ref).sum())} clics postérieurs ignorés")
    snap = recommender.build_snapshot(clicks, art_ids, art_created,
                                      embeddings, t_ref)
    path = os.path.join(args.out, SNAPSHOT_NAME)
    recommender.save_snapshot(snap, path)
    report(path)
    export_app_data(snap, args.data_dir)
    print(f"\nPréparation terminée en {time.time() - start:.0f} s")

    if args.upload:
        upload_snapshot(path, args.container)


if __name__ == "__main__":
    main()

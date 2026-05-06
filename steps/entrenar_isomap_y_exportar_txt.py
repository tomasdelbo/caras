import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from sklearn.manifold import Isomap

from config import (
    CUSTOM_TEST_DIR,
    ISOMAP_DIR,
    ISOMAP_INPUT_DIM,
    ISOMAP_INPUT_SIZE,
    ISOMAP_N_COMPONENTS,
    ISOMAP_N_NEIGHBORS,
    TEST_CUSTOM_TXT,
    TEST_DIR,
    TEST_TXT,
    TRAINING_DIR,
    TRAINING_TXT,
)

log = logging.getLogger(__name__)


def _cargar_grupo(directorio: Path) -> tuple[np.ndarray, list[str], list[Path]]:
    vectores: list[np.ndarray] = []
    etiquetas: list[str] = []
    paths: list[Path] = []
    if not directorio.is_dir():
        return np.empty((0, ISOMAP_INPUT_DIM)), etiquetas, paths

    for sub in sorted(p for p in directorio.iterdir() if p.is_dir()):
        for img_path in sorted(sub.iterdir()):
            if img_path.is_dir() or img_path.name.startswith("."):
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                continue
            small = cv2.resize(img, (ISOMAP_INPUT_SIZE, ISOMAP_INPUT_SIZE))
            vectores.append(small.astype(np.float64).ravel())
            etiquetas.append(sub.name)
            paths.append(img_path)

    if not vectores:
        return np.empty((0, ISOMAP_INPUT_DIM)), etiquetas, paths
    return np.vstack(vectores), etiquetas, paths


def _escribir_embeddings(txt_path: Path, embedding: np.ndarray, etiquetas: list[str]) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    n_comp = embedding.shape[1] if embedding.size else 0
    with open(txt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([f"x{i+1}" for i in range(n_comp)] + ["y"])
        for vec, tag in zip(embedding, etiquetas, strict=True):
            w.writerow([f"{v:.6g}" for v in vec] + [tag])


def entrenar_isomap_y_exportar_txt() -> dict:
    X_train, y_train, paths_train = _cargar_grupo(TRAINING_DIR)
    X_test, y_test, paths_test = _cargar_grupo(TEST_DIR)
    X_custom, y_custom, paths_custom = _cargar_grupo(CUSTOM_TEST_DIR)
    if X_train.shape[0] == 0:
        raise ValueError("No hay imágenes de training. Ejecutá el paso de recorte y split antes.")

    n_neighbors = max(2, min(ISOMAP_N_NEIGHBORS, X_train.shape[0] - 1))
    n_components = min(ISOMAP_N_COMPONENTS, X_train.shape[0] - 1, X_train.shape[1])
    log.info(
        "Ajustando ISOMAP: n_train=%d, n_neighbors=%d, n_components=%d",
        X_train.shape[0], n_neighbors, n_components,
    )
    iso = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric="euclidean")
    emb_train = iso.fit_transform(X_train)
    emb_test = iso.transform(X_test) if X_test.shape[0] > 0 else np.empty((0, n_components))
    emb_custom = iso.transform(X_custom) if X_custom.shape[0] > 0 else np.empty((0, n_components))

    ISOMAP_DIR.mkdir(parents=True, exist_ok=True)
    _escribir_embeddings(TRAINING_TXT, emb_train, y_train)
    _escribir_embeddings(TEST_TXT, emb_test, y_test)
    log.info("Embeddings escritos: %s y %s", TRAINING_TXT, TEST_TXT)
    if X_custom.shape[0] > 0:
        _escribir_embeddings(TEST_CUSTOM_TXT, emb_custom, y_custom)
        log.info("Embeddings custom escritos: %s (n=%d)", TEST_CUSTOM_TXT, X_custom.shape[0])
    else:
        log.info("Sin fotos en %s; no se escribe %s.", CUSTOM_TEST_DIR, TEST_CUSTOM_TXT)

    return {
        "isomap": iso,
        "X_train": X_train,
        "y_train": y_train,
        "paths_train": paths_train,
        "emb_train": emb_train,
        "X_test": X_test,
        "y_test": y_test,
        "paths_test": paths_test,
        "emb_test": emb_test,
        "X_custom": X_custom,
        "y_custom": y_custom,
        "paths_custom": paths_custom,
        "emb_custom": emb_custom,
    }

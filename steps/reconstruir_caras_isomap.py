import logging
from pathlib import Path

import cv2
import numpy as np

from config import (
    ISOMAP_CUSTOM_TEST_DIR,
    ISOMAP_DIR,
    ISOMAP_INPUT_SIZE,
    ISOMAP_TEST_DIR,
    ISOMAP_TRAINING_DIR,
    RECONSTRUCTION_K,
)

log = logging.getLogger(__name__)


def _vecinos_promedio(emb: np.ndarray, emb_train: np.ndarray, X_train: np.ndarray, k: int, excluir_self: bool) -> np.ndarray:
    n = emb.shape[0]
    salida = np.empty((n, X_train.shape[1]), dtype=np.float64)
    for i in range(n):
        diffs = emb_train - emb[i]
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        if excluir_self:
            dists[i] = np.inf
        idx = np.argsort(dists)[:k]
        pesos = 1.0 / (dists[idx] + 1e-9)
        pesos /= pesos.sum()
        salida[i] = (pesos[:, None] * X_train[idx]).sum(axis=0)
    return salida


def _guardar(salida_dir: Path, etiquetas: list[str], paths_origen: list[Path], reconst: np.ndarray) -> None:
    salida_dir.mkdir(parents=True, exist_ok=True)
    for tag, src, vec in zip(etiquetas, paths_origen, reconst, strict=True):
        carpeta = salida_dir / tag
        carpeta.mkdir(parents=True, exist_ok=True)
        img = np.clip(vec, 0, 255).astype(np.uint8).reshape(ISOMAP_INPUT_SIZE, ISOMAP_INPUT_SIZE)
        cv2.imwrite(str(carpeta / src.name), img)


def reconstruir_caras_isomap(ctx: dict) -> None:
    X_train = ctx["X_train"]
    emb_train = ctx["emb_train"]
    paths_train = ctx["paths_train"]
    y_train = ctx["y_train"]

    k = max(1, min(RECONSTRUCTION_K, X_train.shape[0] - 1))
    log.info("Reconstrucción ISOMAP con K=%d vecinos", k)

    rec_train = _vecinos_promedio(emb_train, emb_train, X_train, k=k, excluir_self=True)
    _guardar(ISOMAP_TRAINING_DIR, y_train, paths_train, rec_train)

    if ctx["X_test"].shape[0] > 0:
        rec_test = _vecinos_promedio(ctx["emb_test"], emb_train, X_train, k=k, excluir_self=False)
        _guardar(ISOMAP_TEST_DIR, ctx["y_test"], ctx["paths_test"], rec_test)

    if ctx.get("X_custom") is not None and ctx["X_custom"].shape[0] > 0:
        rec_custom = _vecinos_promedio(ctx["emb_custom"], emb_train, X_train, k=k, excluir_self=False)
        _guardar(ISOMAP_CUSTOM_TEST_DIR, ctx["y_custom"], ctx["paths_custom"], rec_custom)

    log.info("Reconstrucciones guardadas bajo %s", ISOMAP_DIR)

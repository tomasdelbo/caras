import csv
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from config import (
    CARAS_1200_DIR,
    SEED,
    SPLIT_CSV,
    TEST_DIR,
    TEST_RATIO,
    TRAINING_DIR,
)
from utils.extraer_nombre_persona import extraer_nombre_persona
from utils.listar_imagenes_originales import listar_imagenes_originales
from utils.recorte_rostro_dos_ojos import recorte_rostro_dos_ojos

log = logging.getLogger(__name__)


def _reset_caras_1200() -> None:
    if CARAS_1200_DIR.exists():
        shutil.rmtree(CARAS_1200_DIR)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def _guardar_recorte(persona: str, recorte: np.ndarray, raiz: Path, n: int, split: str, writer) -> None:
    carpeta = raiz / persona
    carpeta.mkdir(parents=True, exist_ok=True)
    out_path = carpeta / f"{persona}_{n:03d}.png"
    cv2.imwrite(str(out_path), recorte)
    writer.writerow([split, persona, str(out_path)])


def _split_train_test(n: int, rng: random.Random) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng.shuffle(indices)
    if n < 2:
        return indices, []
    n_test = max(1, round(TEST_RATIO * n))
    n_test = min(n_test, n - 1)
    return sorted(indices[n_test:]), sorted(indices[:n_test])


def recortar_y_partir_train_test() -> None:
    _reset_caras_1200()
    lista = listar_imagenes_originales()
    if not lista:
        raise FileNotFoundError("No hay imágenes en original/.")

    por_persona: dict[str, list[np.ndarray]] = {}
    for path_carga, nombre_original in lista:
        persona = extraer_nombre_persona(nombre_original)
        if not persona:
            continue
        log.info("Procesando: %s -> %s", nombre_original, persona)
        img = cv2.imread(str(path_carga))
        if img is None:
            log.warning("  No se pudo cargar, omitiendo.")
            continue
        recorte = recorte_rostro_dos_ojos(img)
        if recorte is None:
            log.warning("  Sin cara frontal con 2 ojos, omitiendo.")
            continue
        por_persona.setdefault(persona, []).append(recorte)

    if not por_persona:
        raise ValueError("Ninguna foto válida tras recortar caras con 2 ojos.")

    rng = random.Random(SEED)
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "persona", "path"])
        for persona, recortes in sorted(por_persona.items()):
            train_idx, test_idx = _split_train_test(len(recortes), rng)
            log.info("  %s: total=%d train=%d test=%d", persona, len(recortes), len(train_idx), len(test_idx))
            for new_n, i in enumerate(train_idx, 1):
                _guardar_recorte(persona, recortes[i], TRAINING_DIR, new_n, "training", w)
            for new_n, i in enumerate(test_idx, 1):
                _guardar_recorte(persona, recortes[i], TEST_DIR, new_n, "test", w)
    log.info("Split CSV escrito: %s", SPLIT_CSV)

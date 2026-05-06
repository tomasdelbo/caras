import logging
import shutil

import cv2

from config import CUSTOM_TEST_DIR, ORIGINAL_CUSTOM_TEST_DIR
from utils.recorte_rostro_dos_ojos import recorte_rostro_dos_ojos

log = logging.getLogger(__name__)


def recortar_original_custom_test() -> None:
    """Recorta las fotos crudas de original_custom_test/<persona>/ con el mismo
    pipeline que el paso 1 (recorte_rostro_dos_ojos -> 1200x1200 gris) y las
    persiste en caras_1200/custom_test/<persona>/. No hay split: todo lo que
    entra cae en custom_test."""
    if CUSTOM_TEST_DIR.exists():
        shutil.rmtree(CUSTOM_TEST_DIR)

    if not ORIGINAL_CUSTOM_TEST_DIR.is_dir():
        log.info("No existe %s; salteo paso de custom test.", ORIGINAL_CUSTOM_TEST_DIR)
        return

    CUSTOM_TEST_DIR.mkdir(parents=True, exist_ok=True)

    total_validas = 0
    total_descartadas = 0

    for sub in sorted(p for p in ORIGINAL_CUSTOM_TEST_DIR.iterdir() if p.is_dir()):
        persona = sub.name
        validas = 0
        descartadas = 0
        salida_persona = CUSTOM_TEST_DIR / persona
        for img_path in sorted(sub.iterdir()):
            if img_path.is_dir() or img_path.name.startswith("."):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("  [%s] no se pudo leer %s, omitiendo.", persona, img_path.name)
                descartadas += 1
                continue
            recorte = recorte_rostro_dos_ojos(img)
            if recorte is None:
                log.warning("  [%s] %s sin cara frontal con 2 ojos, omitiendo.", persona, img_path.name)
                descartadas += 1
                continue
            salida_persona.mkdir(parents=True, exist_ok=True)
            validas += 1
            out_path = salida_persona / f"{persona}_{validas:03d}.png"
            cv2.imwrite(str(out_path), recorte)
        if validas or descartadas:
            log.info("  %s: validas=%d descartadas=%d", persona, validas, descartadas)
        total_validas += validas
        total_descartadas += descartadas

    log.info(
        "Custom test recortado en %s (validas=%d, descartadas=%d).",
        CUSTOM_TEST_DIR, total_validas, total_descartadas,
    )

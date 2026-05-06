import logging
import sys

from config import SPLIT_CSV
from steps.entrenar_isomap_y_exportar_txt import entrenar_isomap_y_exportar_txt
from steps.recortar_original_custom_test import recortar_original_custom_test
from steps.recortar_y_partir_train_test import recortar_y_partir_train_test
from steps.reconstruir_caras_isomap import reconstruir_caras_isomap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("Paso 1a: recorte 1200x1200 (2 ojos) + split 90/10 train/test desde original/")
    recortar_y_partir_train_test()

    log.info("Paso 1b: recorte 1200x1200 (2 ojos) desde original_custom_test/ -> caras_1200/custom_test/")
    recortar_original_custom_test()

    log.info("Paso 2: ISOMAP 30x30 -> embeddings 70D + TXTs en isomap/")
    ctx = entrenar_isomap_y_exportar_txt()

    log.info("Paso 3: reconstrucción ISOMAP de control")
    reconstruir_caras_isomap(ctx)

    if SPLIT_CSV.exists():
        SPLIT_CSV.unlink()
        log.info("Limpieza: borrado %s", SPLIT_CSV)

    log.info("Pipeline completo.")


if __name__ == "__main__":
    main()

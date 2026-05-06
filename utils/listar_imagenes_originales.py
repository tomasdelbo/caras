import logging
from pathlib import Path

from config import CONVERTIDOS_DIR, ORIGINAL_DIR
from utils.convertir_heic_a_jpeg import convertir_heic_a_jpeg

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}

log = logging.getLogger(__name__)


def listar_imagenes_originales() -> list[tuple[Path, str]]:
    if not ORIGINAL_DIR.is_dir():
        raise FileNotFoundError(f"No existe '{ORIGINAL_DIR}'.")
    archivos = sorted(p for p in ORIGINAL_DIR.iterdir() if p.suffix.lower() in EXTENSIONES)
    heics_sin_cache = [
        p for p in archivos
        if p.suffix.lower() == ".heic" and not (CONVERTIDOS_DIR / (p.stem + ".jpeg")).exists()
    ]
    n_a_convertir = len(heics_sin_cache)
    if n_a_convertir > 0:
        log.info(
            "Convirtiendo %d HEIC -> JPEG (cache en %s, ~0.4s c/u, puede tardar varios minutos)...",
            n_a_convertir, CONVERTIDOS_DIR,
        )

    lista: list[tuple[Path, str]] = []
    convertidos_ok = 0
    heic_idx = 0
    for path in archivos:
        if path.suffix.lower() == ".heic":
            destino = CONVERTIDOS_DIR / (path.stem + ".jpeg")
            if not destino.exists():
                heic_idx += 1
                log.info("  HEIC %d/%d: %s", heic_idx, n_a_convertir, path.name)
                if not convertir_heic_a_jpeg(path, destino):
                    log.warning("  fallo convirtiendo %s, omitiendo.", path.name)
                    continue
                convertidos_ok += 1
            lista.append((destino, path.name))
        else:
            lista.append((path, path.name))

    if n_a_convertir > 0:
        log.info("HEICs convertidos: %d/%d", convertidos_ok, n_a_convertir)
    return lista

from pathlib import Path

from config import CONVERTIDOS_DIR, ORIGINAL_DIR
from utils.convertir_heic_a_jpeg import convertir_heic_a_jpeg

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}


def listar_imagenes_originales() -> list[tuple[Path, str]]:
    if not ORIGINAL_DIR.is_dir():
        raise FileNotFoundError(f"No existe '{ORIGINAL_DIR}'.")
    archivos = sorted(p for p in ORIGINAL_DIR.iterdir() if p.suffix.lower() in EXTENSIONES)
    lista: list[tuple[Path, str]] = []
    for path in archivos:
        if path.suffix.lower() == ".heic":
            destino = CONVERTIDOS_DIR / (path.stem + ".jpeg")
            if not destino.exists() and not convertir_heic_a_jpeg(path, destino):
                continue
            lista.append((destino, path.name))
        else:
            lista.append((path, path.name))
    return lista

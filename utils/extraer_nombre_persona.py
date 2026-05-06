import re
from pathlib import Path

from config import NOMBRE_CANONICO


def extraer_nombre_persona(nombre_archivo: str) -> str:
    stem = Path(nombre_archivo).stem
    sin_parentesis = re.sub(r"\s*\([^)]*\)", "", stem)
    sin_numeros = re.sub(r"\d+$", "", sin_parentesis).rstrip("_ -")
    nombre = sin_numeros.split("_")[0].strip().lower()
    return NOMBRE_CANONICO.get(nombre, nombre)

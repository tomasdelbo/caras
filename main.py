"""
IMAGEN -> recorte rostro 1200x1200 (2 ojos, centrado entre ojos) -> gris -> guardar + CSV temp -> PCA (30x30) -> distancias.
Sin pandas; CSV con módulo csv.
"""

import csv
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

try:
    import pillow_heif
    from PIL import Image
    HAS_HEIC = True
except ImportError:
    HAS_HEIC = False

ORIGINAL_DIR = Path("original")
CARAS_900_DIR = Path("caras_1200")
CONVERTIDOS_DIR = "convertidos"
TEMP_CSV = Path("temp_caras_1200.csv")
PCA_CSV = Path("pca_vectores.csv")
CROP_SIZE = 1200
MARGEN_ROSTRO = 1.2  # recorte ajustado al rostro, poco fondo
PCA_SIZE = 30  # tamaño para PCA (30x30)
N_COMPONENTS = 50

CASCADE_FACE = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
CASCADE_EYES = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
NOMBRE_CANONICO: dict[str, str] = {"migue": "miguel", "juan": "juani"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger(__name__)


def extraer_nombre_persona(nombre_archivo: str) -> str:
    stem = Path(nombre_archivo).stem
    sin_parentesis = re.sub(r"\s*\([^)]*\)", "", stem)
    sin_numeros = re.sub(r"\d+$", "", sin_parentesis).rstrip("_ -")
    nombre = sin_numeros.split("_")[0].strip().lower()
    return NOMBRE_CANONICO.get(nombre, nombre)


def _convertir_heic_a_jpeg(path_heic: Path, path_jpeg: Path) -> bool:
    if not HAS_HEIC:
        return False
    try:
        path_jpeg.parent.mkdir(parents=True, exist_ok=True)
        pillow_heif.register_heif_opener()
        pil_img = Image.open(path_heic)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(str(path_jpeg), bgr)
    except Exception:
        return False


def _cargar_imagen(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path))


def _recorte_rostro_con_dos_ojos(imagen: np.ndarray) -> np.ndarray | None:
    """Recorta un cuadrado que incluye el rostro entero (centrado en la cara), con 2 ojos bien visibles. Devuelve CROP_SIZE x CROP_SIZE en gris."""
    if imagen is None or imagen.size == 0:
        return None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
    face_cascade = cv2.CascadeClassifier(str(CASCADE_FACE))
    eye_cascade = cv2.CascadeClassifier(str(CASCADE_EYES))
    caras = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(caras) == 0:
        return None
    (fx, fy, fw, fh) = max(caras, key=lambda r: r[2] * r[3])
    zona_ojos = gris[fy : fy + int(fh * 0.55), fx : fx + fw]
    ojos = eye_cascade.detectMultiScale(zona_ojos, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    # Centros de ojos en coords de la imagen completa
    ojos_global = [(fx + ox + ow // 2, fy + oy + oh // 2) for (ox, oy, ow, oh) in ojos]
    # Exigir 2 ojos bien diferenciados: altura similar y separados en X (no el mismo ojo ni falsos)
    ojos_global.sort(key=lambda p: p[0])
    pareja_ok = None
    for i in range(len(ojos_global)):
        for j in range(i + 1, len(ojos_global)):
            cx1, cy1 = ojos_global[i]
            cx2, cy2 = ojos_global[j]
            sep_x = abs(cx2 - cx1)
            sep_y = abs(cy2 - cy1)
            if sep_x < fw * 0.15:
                continue  # demasiado juntos, probablemente mismo ojo o falso
            if sep_y > fh * 0.3:
                continue  # altura muy distinta, no son los dos ojos
            if sep_x > fw * 1.1:
                continue  # demasiado separados para ser ojos de la misma cara
            pareja_ok = (ojos_global[i], ojos_global[j])
            break
        if pareja_ok is not None:
            break
    if pareja_ok is None:
        return None
    # Centrar el recorte en el punto medio entre los dos ojos
    (e1x, e1y), (e2x, e2y) = pareja_ok
    center_cx = (e1x + e2x) // 2
    center_cy = (e1y + e2y) // 2
    lado = int(max(fw, fh) * MARGEN_ROSTRO)
    lado = min(lado, imagen.shape[0], imagen.shape[1])
    if lado <= 0:
        return None
    x1 = center_cx - lado // 2
    y1 = center_cy - lado // 2
    x1 = max(0, min(x1, imagen.shape[1] - lado))
    y1 = max(0, min(y1, imagen.shape[0] - lado))
    x2 = x1 + lado
    y2 = y1 + lado
    # Comprobar que ambos ojos quedan dentro del recorte con margen (no cortados)
    margen_borde = int(lado * 0.08)
    for (ex, ey) in pareja_ok:
        if ex < x1 + margen_borde or ex > x2 - margen_borde or ey < y1 + margen_borde or ey > y2 - margen_borde:
            return None
    if x2 > imagen.shape[1] or y2 > imagen.shape[0]:
        return None
    recorte = imagen[y1:y2, x1:x2]
    if recorte.size == 0:
        return None
    gris_rec = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY) if len(recorte.shape) == 3 else recorte
    return cv2.resize(gris_rec, (CROP_SIZE, CROP_SIZE))


def listar_imagenes_originales() -> list[tuple[Path, str]]:
    if not ORIGINAL_DIR.is_dir():
        raise FileNotFoundError(f"No existe '{ORIGINAL_DIR}'.")
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
    archivos = sorted([p for p in ORIGINAL_DIR.iterdir() if p.suffix.lower() in extensiones])
    lista: list[tuple[Path, str]] = []
    cache_dir = ORIGINAL_DIR / CONVERTIDOS_DIR
    for path in archivos:
        if path.suffix.lower() == ".heic":
            path_jpeg = cache_dir / (path.stem + ".jpeg")
            if not path_jpeg.exists():
                log.info("Convirtiendo HEIC -> JPEG: %s", path.name)
                if not _convertir_heic_a_jpeg(path, path_jpeg):
                    continue
            lista.append((path_jpeg, path.name))
        else:
            lista.append((path, path.name))
    return lista


def paso1_recortar_y_temp_csv() -> None:
    """Carga cada foto, recorta rostro entero (2 ojos), guarda en disco y escribe (persona, path) en CSV temp."""
    CARAS_900_DIR.mkdir(exist_ok=True)
    lista = listar_imagenes_originales()
    if not lista:
        raise FileNotFoundError(f"No hay imágenes en {ORIGINAL_DIR}.")
    contador: dict[str, int] = {}
    with open(TEMP_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona", "path"])
        for path_carga, nombre_original in lista:
            persona = extraer_nombre_persona(nombre_original)
            if not persona:
                continue
            log.info("Cargando: %s", nombre_original)
            img = _cargar_imagen(path_carga)
            if img is None:
                log.warning("  -> No se pudo cargar, omitiendo.")
                continue
            log.info("  Recortando rostro entero %dx%d (2 ojos)...", CROP_SIZE, CROP_SIZE)
            recorte = _recorte_rostro_con_dos_ojos(img)
            if recorte is None:
                log.warning("  -> Sin cara frontal con 2 ojos, omitiendo.")
                continue
            contador[persona] = contador.get(persona, 0) + 1
            num = contador[persona]
            (CARAS_900_DIR / persona).mkdir(exist_ok=True)
            out_path = CARAS_900_DIR / persona / f"{persona}_{num:03d}.png"
            cv2.imwrite(str(out_path), recorte)
            log.info("  -> Guardado: %s", out_path)
            w.writerow([persona, str(out_path)])
            del img, recorte  # liberar antes de la siguiente


def paso2_pca_desde_temp_csv() -> tuple[np.ndarray, list[str], list[str]]:
    """Lee temp CSV, carga cada imagen, redimensiona a 30x30 para PCA, escribe pca_vectores.csv. Devuelve (X_pca, personas, paths)."""
    if not TEMP_CSV.exists():
        raise FileNotFoundError(f"Ejecutá antes paso 1. No existe {TEMP_CSV}.")
    personas: list[str] = []
    paths: list[str] = []
    vectores: list[np.ndarray] = []
    with open(TEMP_CSV, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            persona, path_str = row["persona"], row["path"]
            log.info("  [%d] Cargando desde disco: %s", i + 1, path_str)
            img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                log.warning("    -> No se pudo leer, omitiendo.")
                continue
            small = cv2.resize(img, (PCA_SIZE, PCA_SIZE))
            vectores.append(small.astype(np.float64).ravel())
            personas.append(persona)
            paths.append(path_str)
    if not vectores:
        raise ValueError("No hay filas válidas en temp CSV.")
    X = np.vstack(vectores)
    n_comp = min(N_COMPONENTS, X.shape[0] - 1, X.shape[1])
    log.info("Ajustando PCA (n_components=%d)...", n_comp)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    log.info("Guardando vectores en %s", PCA_CSV)
    with open(PCA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona", "path"] + [f"pc{j+1}" for j in range(X_pca.shape[1])])
        for i in range(len(personas)):
            w.writerow([personas[i], paths[i]] + [f"{X_pca[i, j]:.6g}" for j in range(X_pca.shape[1])])
    return X_pca.astype(np.float64), personas, paths


def cargar_pca_desde_csv() -> tuple[np.ndarray, list[str], list[str]]:
    """Carga vectores PCA desde pca_vectores.csv. Devuelve (X_pca, personas, paths)."""
    if not PCA_CSV.exists():
        raise FileNotFoundError(f"No existe {PCA_CSV}. Ejecutá el pipeline completo primero.")
    with open(PCA_CSV, encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise ValueError(f"{PCA_CSV} está vacío.")
    personas = [row["persona"] for row in rows]
    paths = [row["path"] for row in rows]
    pc_cols = [k for k in rows[0].keys() if k.startswith("pc")]
    pc_cols.sort(key=lambda x: int(x[2:]))
    X_pca = np.array([[float(row[c]) for c in pc_cols] for row in rows], dtype=np.float64)
    return X_pca, personas, paths


def paso3_distancias_top(X: np.ndarray, personas: list[str], paths: list[str], top_n: int = 5) -> None:
    """Top N pares de personas con la foto más parecida (menor distancia entre una foto de A y una de B). Muestra enlace a las dos fotos y distancia."""
    D = euclidean_distances(X)
    personas_unicas = sorted(set(personas))
    if len(personas_unicas) < 2:
        log.warning("Se necesitan al menos 2 personas para TOP.")
        return
    resultados: list[tuple[str, str, str, str, float]] = []
    for i, a in enumerate(personas_unicas):
        for b in personas_unicas[i + 1 :]:
            idx_a = [j for j, p in enumerate(personas) if p == a]
            idx_b = [j for j, p in enumerate(personas) if p == b]
            dists = D[np.ix_(idx_a, idx_b)]
            min_flat = np.argmin(dists)
            ia, ib = np.unravel_index(min_flat, dists.shape)
            d = float(dists[ia, ib])
            path_a = paths[idx_a[ia]]
            path_b = paths[idx_b[ib]]
            resultados.append((a, b, path_a, path_b, d))
    resultados.sort(key=lambda x: x[4])
    top = resultados[:top_n]
    print(f"\n--- TOP {top_n} pares de personas con la foto más parecida (distancia mínima) ---")
    for r, (pa, pb, path_a, path_b, dist) in enumerate(top, 1):
        print(f"  {r}. {pa} — {pb}  |  distancia: {dist:.4f}")
        print(f"      foto {pa}: {path_a}")
        print(f"      foto {pb}: {path_b}")


def paso3_distancias_avg(X: np.ndarray, personas: list[str], paths: list[str], top_n: int = 5) -> None:
    """Top N pares de personas más parecidos en promedio (menor distancia promedio entre todas las fotos de A y todas las de B). Muestra valor de referencia."""
    D = euclidean_distances(X)
    personas_unicas = sorted(set(personas))
    if len(personas_unicas) < 2:
        log.warning("Se necesitan al menos 2 personas para AVG.")
        return
    resultados: list[tuple[str, str, float]] = []
    for i, a in enumerate(personas_unicas):
        for b in personas_unicas[i + 1 :]:
            idx_a = [j for j, p in enumerate(personas) if p == a]
            idx_b = [j for j, p in enumerate(personas) if p == b]
            dists = D[np.ix_(idx_a, idx_b)]
            prom = float(np.mean(dists))
            resultados.append((a, b, prom))
    resultados.sort(key=lambda x: x[2])
    top = resultados[:top_n]
    print(f"\n--- TOP {top_n} pares de personas más parecidos (distancia promedio) ---")
    for r, (pa, pb, dist) in enumerate(top, 1):
        print(f"  {r}. {pa} — {pb}  |  distancia promedio: {dist:.4f}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="IMAGEN -> recorte rostro 1200x1200 -> temp CSV -> PCA -> distancias")
    p.add_argument("--modo", choices=["top", "avg"], default="top")
    p.add_argument("--solo-crop", action="store_true", help="Solo recortar y escribir temp CSV")
    args = p.parse_args()

    if args.solo_crop:
        log.info("Paso 1: Recortar rostro %dx%d y escribir %s", CROP_SIZE, CROP_SIZE, TEMP_CSV)
        paso1_recortar_y_temp_csv()
        log.info("Listo. Temp CSV: %s", TEMP_CSV)
        return

    if PCA_CSV.exists():
        log.info("PCA ya existe (%s), cargando y calculando distancias.", PCA_CSV)
        X_pca, personas, paths = cargar_pca_desde_csv()
    else:
        log.info("Paso 1: Recortar rostro %dx%d y escribir %s", CROP_SIZE, CROP_SIZE, TEMP_CSV)
        paso1_recortar_y_temp_csv()
        log.info("Paso 2: PCA desde temp CSV -> %s", PCA_CSV)
        X_pca, personas, paths = paso2_pca_desde_temp_csv()

    log.info("Paso 3: Distancias (%s)", args.modo)
    if args.modo == "top":
        paso3_distancias_top(X_pca, personas, paths, top_n=5)
    else:
        paso3_distancias_avg(X_pca, personas, paths, top_n=5)


if __name__ == "__main__":
    main()

from pathlib import Path

import cv2
import numpy as np

try:
    import pillow_heif
    from PIL import Image
    _HEIC_OK = True
except ImportError:
    _HEIC_OK = False


def convertir_heic_a_jpeg(path_heic: Path, path_jpeg: Path) -> bool:
    if not _HEIC_OK:
        return False
    try:
        path_jpeg.parent.mkdir(parents=True, exist_ok=True)
        pillow_heif.register_heif_opener()
        pil_img = Image.open(path_heic)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return cv2.imwrite(str(path_jpeg), bgr)
    except Exception:
        return False

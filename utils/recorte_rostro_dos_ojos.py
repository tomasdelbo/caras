import cv2
import numpy as np

from config import CASCADE_EYES, CASCADE_FACE, CROP_SIZE, MARGEN_ROSTRO


def recorte_rostro_dos_ojos(imagen: np.ndarray) -> np.ndarray | None:
    if imagen is None or imagen.size == 0:
        return None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
    face_cascade = cv2.CascadeClassifier(str(CASCADE_FACE))
    eye_cascade = cv2.CascadeClassifier(str(CASCADE_EYES))
    caras = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(caras) == 0:
        return None
    fx, fy, fw, fh = max(caras, key=lambda r: r[2] * r[3])
    zona_ojos = gris[fy : fy + int(fh * 0.55), fx : fx + fw]
    ojos = eye_cascade.detectMultiScale(zona_ojos, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))
    ojos_global = sorted(
        ((fx + ox + ow // 2, fy + oy + oh // 2) for (ox, oy, ow, oh) in ojos),
        key=lambda p: p[0],
    )
    pareja_ok = None
    for i in range(len(ojos_global)):
        for j in range(i + 1, len(ojos_global)):
            cx1, cy1 = ojos_global[i]
            cx2, cy2 = ojos_global[j]
            sep_x = abs(cx2 - cx1)
            sep_y = abs(cy2 - cy1)
            if sep_x < fw * 0.15 or sep_y > fh * 0.3 or sep_x > fw * 1.1:
                continue
            pareja_ok = (ojos_global[i], ojos_global[j])
            break
        if pareja_ok is not None:
            break
    if pareja_ok is None:
        return None
    (e1x, e1y), (e2x, e2y) = pareja_ok
    center_cx = (e1x + e2x) // 2
    center_cy = (e1y + e2y) // 2
    lado = int(max(fw, fh) * MARGEN_ROSTRO)
    lado = min(lado, imagen.shape[0], imagen.shape[1])
    if lado <= 0:
        return None
    x1 = max(0, min(center_cx - lado // 2, imagen.shape[1] - lado))
    y1 = max(0, min(center_cy - lado // 2, imagen.shape[0] - lado))
    x2 = x1 + lado
    y2 = y1 + lado
    margen_borde = int(lado * 0.08)
    for ex, ey in pareja_ok:
        if ex < x1 + margen_borde or ex > x2 - margen_borde or ey < y1 + margen_borde or ey > y2 - margen_borde:
            return None
    if x2 > imagen.shape[1] or y2 > imagen.shape[0]:
        return None
    recorte = imagen[y1:y2, x1:x2]
    if recorte.size == 0:
        return None
    gris_rec = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY) if len(recorte.shape) == 3 else recorte
    return cv2.resize(gris_rec, (CROP_SIZE, CROP_SIZE))

# DMA – TP Caras + PCA

Pipeline: fotos en `original/` → recorte rostro 1200×1200 (2 ojos, escala de grises) en `caras_1200/` → redimensionar a 30×30 (900 píxeles por imagen) → **PCA** (reduce la dimensión del vector a `pc1`, `pc2`, …) → distancias (TOP 2 más cercanas o AVG).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

1. Colocá las fotos en la carpeta **`original/`**. Se usa solo el nombre (primera parte antes de `_`), en minúsculas (ej. `Juan_Perez_01.jpg` → persona `juan`).

2. Ejecutá el pipeline desde la carpeta `caras/`:
   ```bash
   cd caras
   python main.py
   ```
   - **Paso 1:** Detecta caras frontales con 2 ojos (OpenCV), recorta 1200×1200 en gris, guarda en `caras_1200/<persona>/` y escribe `temp_caras_1200.csv`.
   - **Paso 2:** Carga las imágenes, redimensiona a 30×30 (solo achica la imagen; cada cara queda como vector de 900 valores), aplica **PCA** para comprimir a pocas componentes y guarda esos vectores en `pca_vectores.csv`.
   - **Paso 3:** Calcula distancias y muestra resultados.

3. **Opciones:**
   - **`--modo top`** (por defecto): para cada foto muestra las **2 fotos más cercanas** (nombre + distancia).
   - **`--modo avg`**: para cada foto muestra la **distancia promedio** al resto.
   - **`--solo-crop`**: solo ejecuta paso 1 (recortar y escribir CSV temporal), sin PCA ni distancias.

   ```bash
   python main.py --modo avg
   python main.py --solo-crop
   ```

## Archivos generados (en `.gitignore`)

- `caras_1200/` — recortes 1200×1200 en gris por persona.
- `temp_caras_1200.csv` — lista (persona, path) para no cargar todo en memoria.
- `pca_vectores.csv` — salida del PCA: persona, path y coeficientes `pc1`, `pc2`, … (no son los 900 píxeles del 30×30).
- `convertidos/` — cache de HEIC convertidos a JPEG.

# DMA – TP Caras + ISOMAP

Pipeline: fotos en `original/` → recorte de rostro 1200×1200 (2 ojos, escala de grises) → split 90/10 train/test por persona → redimensionar a 30×30 (vector de 900 valores) → **ISOMAP** (distancia geodésica, 70 componentes ordenadas por importancia) → archivos `.txt` separados por tab `x1..x70,y` + reconstrucciones de control + entrenamiento de una red neuronal con **backpropagation manual** (`examen.py`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

## Uso

1. **Descargá las fotos crudas** (no están en el repo por su tamaño) desde el SharePoint del curso y descomprimilas dentro de **`caras/original/`**:

   <https://alumniiaeedu-my.sharepoint.com/shared?listurl=https%3A%2F%2Falumniiaeedu%2Dmy%2Esharepoint%2Ecom%2Fpersonal%2Fmceriotti%5Fmail%5Faustral%5Fedu%5Far%2FDocuments&id=%2Fpersonal%2Fmceriotti%5Fmail%5Faustral%5Fedu%5Far%2FDocuments%2FCaras&viewid=71522521%2Dbbec%2D4825%2D9802%2D8b3b07729c50>

   Las fotos van **sueltas** dentro de `original/` (no en subcarpetas). La etiqueta de persona se toma del nombre del archivo (primera parte antes de `_`, en minúsculas: `Juan_Perez_01.jpg` → `juan`).

2. (Opcional) **Test manual**: dropeá fotos crudas (jpg/png) dentro de **`original_custom_test/<persona>/`** (una carpeta por persona, ya creadas vacías). Pasan exactamente por el mismo flujo que `original/` (recorte 1200×1200 → ISOMAP), pero **sin** participar del split: van todas a un grupo aparte `caras_1200/custom_test/<persona>/` y al TXT `isomap/test_custom_embeddings.txt`. Para probar fotos de personas **no conocidas** (que no están en training), usá la subcarpeta `original_custom_test/otros/` y van con `y=otros`. Sirve para probar fotos nuevas sin tocar el split de training/test del pipeline.

3. Ejecutá el pipeline completo desde la carpeta `caras/`:

   ```bash
   cd caras
   python procesar.py
   ```

   - **Paso 1a (`steps/recortar_y_partir_train_test.py`):** detecta caras frontales con 2 ojos (OpenCV) sobre `original/`, recorta 1200×1200 en gris y reparte cada persona **90 % training / 10 % test** (semilla fija). Escribe en `caras_1200/training/<persona>/`, `caras_1200/test/<persona>/` y `temp_caras_split.csv`.
   - **Paso 1b (`steps/recortar_original_custom_test.py`):** mismo recorte pero sobre `original_custom_test/<persona>/`, sin split. Escribe en `caras_1200/custom_test/<persona>/`.
   - **Paso 2 (`steps/entrenar_isomap_y_exportar_txt.py`):** carga las imágenes de train/test/custom_test, redimensiona a 30×30 (vector de 900 valores), ajusta **ISOMAP** sobre training (grafo de vecinos + caminos geodésicos, 70 componentes), proyecta test y custom_test con `iso.transform()`, y escribe `isomap/training_embeddings.txt`, `isomap/test_embeddings.txt` y (si hay fotos custom) `isomap/test_custom_embeddings.txt`, **separados por tab** con cabecera `x1..x70,y`.
   - **Paso 3 (`steps/reconstruir_caras_isomap.py`):** para cada muestra promedia los **K vecinos más cercanos en el embedding 70D** sobre los vectores 900D de training y guarda PNGs 30×30 en `isomap/training/<persona>/`, `isomap/test/<persona>/` y `isomap/custom_test/<persona>/` como control visual (no es inversa exacta de ISOMAP).

4. Entrená la red neuronal con backpropagation manual:

   ```bash
   python examen.py
   ```

   1. **Training**: predicción fila por fila + accuracy de training.
   2. **Test crudo**: idem sobre `isomap/test_embeddings.txt` (usa el mismo `scaler`, sólo `transform`) + accuracy.
   3. **Test con umbral de confianza** (`UMBRAL_CONFIANZA = 0.75`): si la confianza del argmax es menor al umbral, la predicción se **DESCARTA**. Imprime un resumen con descartados / aceptados / aciertos / errores. La idea es no clasificar caras que la red no reconoce con seguridad (útil para `otros`).

## Estructura de carpetas resultante

```
caras/
├── procesar.py              # orquesta los pasos del pipeline
├── examen.py                # backpropagation manual sobre los embeddings
├── config.py                # paths, constantes e hiperparámetros
├── steps/
│   ├── recortar_y_partir_train_test.py
│   ├── recortar_original_custom_test.py
│   ├── entrenar_isomap_y_exportar_txt.py
│   └── reconstruir_caras_isomap.py
├── utils/
│   ├── extraer_nombre_persona.py
│   ├── convertir_heic_a_jpeg.py
│   ├── listar_imagenes_originales.py
│   └── recorte_rostro_dos_ojos.py
├── original/                       # entrada principal (etiqueta = nombre de archivo)
├── original_custom_test/           # bandeja de fotos crudas para test manual
│   ├── <persona>/                  # con etiqueta conocida (mide accuracy)
│   └── otros/                      # personas no conocidas (y=otros)
├── caras_1200/
│   ├── training/<persona>/         # del split 90% (paso 1a)
│   ├── test/<persona>/             # del split 10% (paso 1a)
│   └── custom_test/<persona>/      # de original_custom_test/ (paso 1b)
└── isomap/
    ├── training_embeddings.txt        # x1\tx2\t...\tx70\ty
    ├── test_embeddings.txt
    ├── test_custom_embeddings.txt     # generado solo si hay fotos en custom_test/
    ├── training/<persona>/            # reconstrucciones 30×30
    ├── test/<persona>/
    └── custom_test/<persona>/
```

## Hiperparámetros del pipeline (en `config.py`)

- `TEST_RATIO = 0.10`, `SEED = 42`
- `ISOMAP_INPUT_SIZE = 30` → vector de **900** features por imagen
- `ISOMAP_N_COMPONENTS = 70` (clipa a `n_train - 1` si hay pocas imágenes)
- `ISOMAP_N_NEIGHBORS = 8`
- `RECONSTRUCTION_K = 5`

## Hiperparámetros de la red (en `examen.py`)

- `np.random.seed = 999979`
- Arquitectura `[32, 16, n_clases]`, todas `logsig`
- `learning_rate = 0.05`
- `epoch_limit = 2500`, `error_delta_umbral = 1e-6`
- `UMBRAL_CONFIANZA = 0.75` (umbral de confianza para descartar predicciones inseguras en test)

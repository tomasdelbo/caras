from pathlib import Path

import cv2

SEED = 999979

ORIGINAL_DIR = Path("original")
CONVERTIDOS_DIR = ORIGINAL_DIR / "convertidos"
ORIGINAL_CUSTOM_TEST_DIR = Path("original_custom_test")

CARAS_1200_DIR = Path("caras_1200")
TRAINING_DIR = CARAS_1200_DIR / "training"
TEST_DIR = CARAS_1200_DIR / "test"
CUSTOM_TEST_DIR = CARAS_1200_DIR / "custom_test"

ISOMAP_DIR = Path("isomap")
ISOMAP_TRAINING_DIR = ISOMAP_DIR / "training"
ISOMAP_TEST_DIR = ISOMAP_DIR / "test"
ISOMAP_CUSTOM_TEST_DIR = ISOMAP_DIR / "custom_test"

TRAINING_TXT = ISOMAP_DIR / "training_embeddings.txt"
TEST_TXT = ISOMAP_DIR / "test_embeddings.txt"
TEST_CUSTOM_TXT = ISOMAP_DIR / "test_custom_embeddings.txt"
SPLIT_CSV = Path("temp_caras_split.csv")

CROP_SIZE = 1200
MARGEN_ROSTRO = 1.2

ISOMAP_INPUT_SIZE = 30
ISOMAP_INPUT_DIM = ISOMAP_INPUT_SIZE * ISOMAP_INPUT_SIZE
ISOMAP_N_COMPONENTS = 90
ISOMAP_N_NEIGHBORS = 8
RECONSTRUCTION_K = 5

TEST_RATIO = 0.10

CASCADE_FACE = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
CASCADE_EYES = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"

NOMBRE_CANONICO: dict[str, str] = {"migue": "miguel", "juan": "juani"}

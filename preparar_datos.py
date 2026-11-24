
from pathlib import Path
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from keras.applications.vgg19 import preprocess_input

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset"

CATEGORIAS = {
    "POLY": 0,
    "NORM": 1
}

TAMANO = (150, 150)

# ==========================================
# 2. CARGAR IMÁGENES
# ==========================================

def cargar_imagenes(carpeta):

    imagenes = []
    etiquetas = []

    for categoria, etiqueta in CATEGORIAS.items():

        ruta = DATASET / carpeta / categoria

        archivos = sorted(ruta.glob("*.jpg"))

        print(f"Cargando {carpeta}/{categoria}: {len(archivos)} imágenes")

        for archivo in archivos:

            imagen = cv2.imread(str(archivo))

            if imagen is None:
                print(f"No se pudo leer: {archivo}")
                continue

            # OpenCV utiliza BGR; convertir a RGB
            imagen = cv2.cvtColor(
                imagen, cv2.COLOR_BGR2RGB
            )

            # Redimensionar
            imagen = cv2.resize(imagen, TAMANO)

            imagenes.append(imagen)
            etiquetas.append(etiqueta)

    X = np.asarray(imagenes, dtype=np.float32)
    Y = np.asarray(etiquetas, dtype=np.int32)

    # Preprocesamiento específico de VGG19
    X = preprocess_input(X)

    return X, Y


# ==========================================
# 3. CARGAR TRAIN
# ==========================================

X, Y = cargar_imagenes("train")

# ==========================================
# 4. DIVIDIR ENTRENAMIENTO Y VALIDACIÓN
# ==========================================

x_train, x_val, y_train, y_val = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

# ==========================================
# 5. MOSTRAR RESULTADOS
# ==========================================

print("\nRESULTADOS DE LA PREPARACIÓN")
print("-" * 45)

print("Imágenes de entrenamiento:", len(x_train))
print("Imágenes de validación:", len(x_val))

print("\nDistribución de entrenamiento:")
print("POLY:", np.sum(y_train == 0))
print("NORM:", np.sum(y_train == 1))

print("\nDistribución de validación:")
print("POLY:", np.sum(y_val == 0))
print("NORM:", np.sum(y_val == 1))

print("\nDimensiones de las imágenes:")
print(x_train.shape)

print("\nPreparación completada correctamente.")

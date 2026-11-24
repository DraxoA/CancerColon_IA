
from pathlib import Path
import time
import json
import random

import cv2
import numpy as np
import tensorflow as tf

from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from modelo_mobilenet import crear_modelo_mobilenet

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset"

MODELOS = BASE_DIR / "modelos"
MODELOS.mkdir(exist_ok=True)

RESULTADOS = BASE_DIR / "resultados" / "mobilenetv2"
RESULTADOS.mkdir(parents=True, exist_ok=True)

RUTA_MODELO = MODELOS / "mobilenetv2_colon.keras"
RUTA_HISTORIAL = RESULTADOS / "historial.json"

CATEGORIAS = {
    "POLY": 0,
    "NORM": 1
}

# ==========================================
# 2. CARGAR IMÁGENES
# ==========================================

def cargar_imagenes():

    imagenes = []
    etiquetas = []

    for categoria, etiqueta in CATEGORIAS.items():

        ruta = DATASET / "train" / categoria
        archivos = sorted(ruta.glob("*.jpg"))

        print(
            f"Cargando {categoria}: {len(archivos)} imágenes"
        )

        for archivo in archivos:

            imagen = cv2.imread(str(archivo))

            if imagen is None:
                raise ValueError(
                    f"No se pudo leer: {archivo}"
                )

            imagen = cv2.cvtColor(
                imagen,
                cv2.COLOR_BGR2RGB
            )

            imagen = cv2.resize(
                imagen,
                (150, 150)
            )

            imagenes.append(imagen)
            etiquetas.append(etiqueta)

    X = np.asarray(imagenes, dtype=np.float32)
    Y = np.asarray(etiquetas, dtype=np.int32)

    # Preprocesamiento de MobileNetV2
    X = preprocess_input(X)

    return X, Y


# ==========================================
# 3. PREPARAR ENTRENAMIENTO Y VALIDACIÓN
# ==========================================

print("\nPreparando imágenes...")

X, Y = cargar_imagenes()

x_train, x_val, y_train, y_val = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print("\nEntrenamiento:", len(x_train))
print("Validación:", len(x_val))

print("POLY entrenamiento:", np.sum(y_train == 0))
print("NORM entrenamiento:", np.sum(y_train == 1))

print("POLY validación:", np.sum(y_val == 0))
print("NORM validación:", np.sum(y_val == 1))

# ==========================================
# 4. CONSTRUIR MODELO
# ==========================================

print("\nConstruyendo MobileNetV2...")

modelo = crear_modelo_mobilenet()

# ==========================================
# 5. CONFIGURAR CALLBACKS
# ==========================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=str(RUTA_MODELO),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# ==========================================
# 6. ENTRENAMIENTO
# ==========================================

print("\nIniciando entrenamiento MobileNetV2...")

inicio = time.time()

historial = modelo.fit(
    x_train,
    y_train,
    batch_size=8,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)

tiempo_total = time.time() - inicio

# ==========================================
# 7. GUARDAR RESULTADOS
# ==========================================

with open(
    RUTA_HISTORIAL,
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(
        historial.history,
        archivo,
        indent=4
    )

print("\nENTRENAMIENTO MOBILENETV2 FINALIZADO")
print("=" * 45)

print(f"Tiempo: {tiempo_total:.2f} segundos")

print(
    "Épocas ejecutadas:",
    len(historial.history["loss"])
)

print(
    "Mejor val_accuracy:",
    max(historial.history["val_accuracy"])
)

print(
    "Mejor val_loss:",
    min(historial.history["val_loss"])
)

print(f"Modelo guardado: {RUTA_MODELO}")
print(f"Historial guardado: {RUTA_HISTORIAL}")

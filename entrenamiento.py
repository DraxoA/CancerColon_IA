
from pathlib import Path
import time
import runpy
import json

from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from modelo import crear_modelo

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CARPETA_MODELOS = BASE_DIR / "modelos"
CARPETA_MODELOS.mkdir(exist_ok=True)

RUTA_MODELO = CARPETA_MODELOS / "vgg19_colon.keras"
RUTA_HISTORIAL = CARPETA_MODELOS / "historial.json"

# ==========================================
# 2. CARGAR DATOS
# ==========================================

print("\nPreparando imágenes...")

datos = runpy.run_path(
    str(BASE_DIR / "preparar_datos.py")
)

x_train = datos["x_train"]
y_train = datos["y_train"]

x_val = datos["x_val"]
y_val = datos["y_val"]

# ==========================================
# 3. CONSTRUIR MODELO
# ==========================================

print("\nConstruyendo modelo VGG19...")

modelo = crear_modelo()

# ==========================================
# 4. CONFIGURAR CALLBACKS
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
# 5. ENTRENAMIENTO
# ==========================================

print("\nIniciando entrenamiento completo...")

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

fin = time.time()

# ==========================================
# 6. GUARDAR HISTORIAL
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

# ==========================================
# 7. RESULTADOS
# ==========================================

print("\nENTRENAMIENTO FINALIZADO")
print("-" * 45)

print(f"Tiempo total: {fin - inicio:.2f} segundos")

print(f"Épocas ejecutadas: {len(historial.history['loss'])}")

print(f"Mejor val_accuracy: {max(historial.history['val_accuracy']):.4f}")

print(f"Mejor val_loss: {min(historial.history['val_loss']):.4f}")

print(f"\nModelo guardado en: {RUTA_MODELO}")

print(f"Historial guardado en: {RUTA_HISTORIAL}")

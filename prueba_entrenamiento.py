
import time
import numpy as np
import runpy

from modelo import crear_modelo

# ==========================================
# 1. CARGAR DATOS PREPARADOS
# ==========================================

print("\nCargando imágenes...")

datos = runpy.run_path("preparar_datos.py")

x_train = datos["x_train"]
y_train = datos["y_train"]

x_val = datos["x_val"]
y_val = datos["y_val"]

# ==========================================
# 2. SELECCIONAR MUESTRA PEQUEÑA
# ==========================================

# Seleccionar 32 imágenes de cada categoría
indices_train = np.concatenate([
    np.where(y_train == 0)[0][:32],
    np.where(y_train == 1)[0][:32]
])

# Seleccionar 8 imágenes de cada categoría
indices_val = np.concatenate([
    np.where(y_val == 0)[0][:8],
    np.where(y_val == 1)[0][:8]
])

x_train_prueba = x_train[indices_train]
y_train_prueba = y_train[indices_train]

x_val_prueba = x_val[indices_val]
y_val_prueba = y_val[indices_val]

print("\nImágenes de entrenamiento:", len(x_train_prueba))
print("Imágenes de validación:", len(x_val_prueba))

# ==========================================
# 3. CREAR MODELO
# ==========================================

print("\nConstruyendo VGG19...")

modelo = crear_modelo()

# ==========================================
# 4. ENTRENAMIENTO DE PRUEBA
# ==========================================

print("\nIniciando prueba de entrenamiento...")

inicio = time.time()

historial = modelo.fit(
    x_train_prueba,
    y_train_prueba,
    batch_size=8,
    epochs=1,
    validation_data=(
        x_val_prueba,
        y_val_prueba
    ),
    verbose=1
)

fin = time.time()

# ==========================================
# 5. RESULTADOS
# ==========================================

print("\nPRUEBA FINALIZADA")
print("-" * 40)

print(f"Tiempo total: {fin - inicio:.2f} segundos")

print(
    "Accuracy entrenamiento:",
    historial.history["accuracy"][-1]
)

print(
    "Accuracy validación:",
    historial.history["val_accuracy"][-1]
)

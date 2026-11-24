
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from keras.applications.vgg19 import preprocess_input

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

RUTA_CC = BASE_DIR / "dataset" / "CC"
RUTA_MODELO = BASE_DIR / "modelos" / "vgg19_colon.keras"

CLASES = ["POLY", "NORM"]
TAMANO = (150, 150)

# ==========================================
# 2. CARGAR IMÁGENES DE CC
# ==========================================

def cargar_cc():

    imagenes = []
    etiquetas = []
    rutas = []

    for etiqueta, categoria in enumerate(CLASES):

        ruta = RUTA_CC / categoria
        archivos = sorted(ruta.glob("*.jpg"))

        print(f"Cargando CC/{categoria}: {len(archivos)}")

        for archivo in archivos:

            imagen = cv2.imread(str(archivo))

            if imagen is None:
                raise ValueError(
                    f"No se pudo leer: {archivo}"
                )

            imagen = cv2.cvtColor(
                imagen, cv2.COLOR_BGR2RGB
            )

            imagen = cv2.resize(imagen, TAMANO)

            imagenes.append(imagen)
            etiquetas.append(etiqueta)
            rutas.append(archivo)

    X = np.asarray(imagenes, dtype=np.float32)
    Y = np.asarray(etiquetas, dtype=np.int32)

    X = preprocess_input(X)

    return X, Y, rutas


# ==========================================
# 3. CARGAR MODELO
# ==========================================

print("\nCargando modelo entrenado...")

modelo = load_model(RUTA_MODELO)

print("\nPreparando imágenes de evaluación...")

x_test, y_test, rutas_test = cargar_cc()

# ==========================================
# 4. REALIZAR PREDICCIONES
# ==========================================

print("\nEvaluando las imágenes de CC...")

probabilidades = modelo.predict(
    x_test,
    batch_size=8,
    verbose=1
)

y_pred = np.argmax(probabilidades, axis=1)

# ==========================================
# 5. CALCULAR MÉTRICAS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nRESULTADOS DE EVALUACIÓN")
print("=" * 45)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy (%): {accuracy * 100:.2f}%")

print("\nREPORTE DE CLASIFICACIÓN")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=CLASES,
        zero_division=0
    )
)

# ==========================================
# 6. MATRIZ DE CONFUSIÓN
# ==========================================

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nMATRIZ DE CONFUSIÓN")
print(matriz)

plt.figure(figsize=(8, 6))

sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASES,
    yticklabels=CLASES
)

plt.xlabel("Predicción del modelo")
plt.ylabel("Categoría real")
plt.title("Matriz de confusión - VGG19")

plt.tight_layout()

RUTA_GRAFICA = (
    BASE_DIR / "modelos" / "matriz_confusion.png"
)

plt.savefig(RUTA_GRAFICA, dpi=300)

plt.show()

# ==========================================
# 7. ERRORES EN DETECCIÓN DE PÓLIPOS
# ==========================================

poly_como_norm = matriz[0, 1]

print("\nANÁLISIS DE PÓLIPOS")
print("=" * 45)

print(
    "Pólipos clasificados como normales:",
    poly_como_norm
)

print("\nEvaluación finalizada.")


# ==========================================
# 8. IDENTIFICAR FALSOS NEGATIVOS
# ==========================================

import shutil
import csv

CARPETA_ERRORES = (
    BASE_DIR / "resultados" / "falsos_negativos"
)

CARPETA_ERRORES.mkdir(
    parents=True,
    exist_ok=True
)

# POLY = 0
# NORM = 1
#
# Falso negativo:
# La imagen realmente es POLY,
# pero el modelo predice NORM.

indices_fn = np.where(
    (y_test == 0) & (y_pred == 1)
)[0]

print("\nFALSOS NEGATIVOS ENCONTRADOS")
print("=" * 50)

print(f"Cantidad: {len(indices_fn)}")

registros = []

for numero, indice in enumerate(indices_fn, start=1):

    ruta_original = rutas_test[indice]

    prob_poly = float(probabilidades[indice][0])
    prob_norm = float(probabilidades[indice][1])

    print(f"\nFalso negativo {numero}")
    print(f"Imagen: {ruta_original.name}")
    print(f"Ruta: {ruta_original}")
    print(f"Categoría real: POLY")
    print(f"Predicción: NORM")
    print(f"Puntuación POLY: {prob_poly:.4f}")
    print(f"Puntuación NORM: {prob_norm:.4f}")

    # Copiar imagen original
    destino = (
        CARPETA_ERRORES
        / f"FN_{numero}_{ruta_original.name}"
    )

    shutil.copy2(ruta_original, destino)

    # Guardar información
    registros.append({
        "imagen": ruta_original.name,
        "ruta_original": str(ruta_original),
        "categoria_real": "POLY",
        "prediccion": "NORM",
        "puntuacion_poly": prob_poly,
        "puntuacion_norm": prob_norm
    })

# ==========================================
# 9. GUARDAR REPORTE CSV
# ==========================================

ruta_csv = (
    CARPETA_ERRORES / "reporte_falsos_negativos.csv"
)

with open(
    ruta_csv,
    "w",
    newline="",
    encoding="utf-8-sig"
) as archivo:

    campos = [
        "imagen",
        "ruta_original",
        "categoria_real",
        "prediccion",
        "puntuacion_poly",
        "puntuacion_norm"
    ]

    escritor = csv.DictWriter(
        archivo,
        fieldnames=campos
    )

    escritor.writeheader()
    escritor.writerows(registros)

print("\nAnálisis de falsos negativos completado.")
print(f"Imágenes guardadas en: {CARPETA_ERRORES}")
print(f"Reporte CSV: {ruta_csv}")

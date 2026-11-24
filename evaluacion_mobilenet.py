
from pathlib import Path
import json
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

RUTA_CC = BASE_DIR / "dataset" / "CC"

RUTA_MODELO = (
    BASE_DIR / "modelos" / "mobilenetv2_colon.keras"
)

RESULTADOS = BASE_DIR / "resultados" / "mobilenetv2"
RESULTADOS.mkdir(parents=True, exist_ok=True)

CLASES = ["POLY", "NORM"]

# ==========================================
# 2. CARGAR IMÁGENES
# ==========================================

def cargar_cc():

    imagenes = []
    etiquetas = []
    rutas = []

    for etiqueta, categoria in enumerate(CLASES):

        carpeta = RUTA_CC / categoria
        archivos = sorted(carpeta.glob("*.jpg"))

        print(f"CC/{categoria}: {len(archivos)} imágenes")

        for archivo in archivos:

            imagen = cv2.imread(str(archivo))

            if imagen is None:
                raise ValueError(
                    f"No se pudo leer: {archivo}"
                )

            imagen = cv2.cvtColor(
                imagen, cv2.COLOR_BGR2RGB
            )

            imagen = cv2.resize(imagen, (150, 150))

            imagenes.append(imagen)
            etiquetas.append(etiqueta)
            rutas.append(archivo)

    X = np.asarray(imagenes, dtype=np.float32)
    Y = np.asarray(etiquetas, dtype=np.int32)

    X = preprocess_input(X)

    return X, Y, rutas


# ==========================================
# 3. CARGAR MODELO Y PREDECIR
# ==========================================

print("\nCargando MobileNetV2...")

modelo = load_model(RUTA_MODELO)

x_test, y_test, rutas = cargar_cc()

print("\nIniciando evaluación...")

probabilidades = modelo.predict(
    x_test,
    batch_size=8,
    verbose=1
)

y_pred = np.argmax(probabilidades, axis=1)

# ==========================================
# 4. CALCULAR MÉTRICAS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

# POLY es la clase positiva
vp = int(matriz[0, 0])
fn = int(matriz[0, 1])
fp = int(matriz[1, 0])
vn = int(matriz[1, 1])

precision = vp / (vp + fp) if (vp + fp) else 0
sensibilidad = vp / (vp + fn) if (vp + fn) else 0
especificidad = vn / (vn + fp) if (vn + fp) else 0

f1 = (
    2 * precision * sensibilidad / (precision + sensibilidad)
    if (precision + sensibilidad) else 0
)

y_binario = (y_test == 0).astype(int)
p_poly = probabilidades[:, 0]

auc_roc = roc_auc_score(y_binario, p_poly)
ap = average_precision_score(y_binario, p_poly)

print("\nRESULTADOS MOBILENETV2")
print("=" * 45)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precisión POLY: {precision * 100:.2f}%")
print(f"Sensibilidad POLY: {sensibilidad * 100:.2f}%")
print(f"Especificidad: {especificidad * 100:.2f}%")
print(f"F1-score POLY: {f1 * 100:.2f}%")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"Average Precision: {ap:.4f}")

print("\nMATRIZ DE CONFUSIÓN")
print(matriz)

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
# 5. GUARDAR RESULTADOS
# ==========================================

metricas = {
    "modelo": "MobileNetV2",
    "imagenes_evaluadas": len(y_test),
    "accuracy": float(accuracy),
    "precision_poly": float(precision),
    "sensibilidad_poly": float(sensibilidad),
    "especificidad": float(especificidad),
    "f1_poly": float(f1),
    "auc_roc": float(auc_roc),
    "average_precision": float(ap),
    "verdaderos_positivos": vp,
    "verdaderos_negativos": vn,
    "falsos_positivos": fp,
    "falsos_negativos": fn
}

with open(
    RESULTADOS / "metricas.json",
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(metricas, archivo, indent=4)

# Guardar todas las predicciones
with open(
    RESULTADOS / "predicciones.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as archivo:

    escritor = csv.writer(archivo)

    escritor.writerow([
        "imagen",
        "categoria_real",
        "categoria_predicha",
        "puntuacion_poly",
        "puntuacion_norm",
        "correcta"
    ])

    for i, ruta in enumerate(rutas):

        escritor.writerow([
            ruta.name,
            CLASES[y_test[i]],
            CLASES[y_pred[i]],
            float(probabilidades[i, 0]),
            float(probabilidades[i, 1]),
            bool(y_test[i] == y_pred[i])
        ])

# ==========================================
# 6. MATRIZ DE CONFUSIÓN
# ==========================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASES,
    yticklabels=CLASES
)

plt.title("Matriz de confusión - MobileNetV2")
plt.xlabel("Categoría predicha")
plt.ylabel("Categoría real")
plt.tight_layout()

plt.savefig(
    RESULTADOS / "matriz_confusion.png",
    dpi=300
)

plt.close()

# ==========================================
# 7. IDENTIFICAR FALSOS NEGATIVOS
# ==========================================

print("\nFALSOS NEGATIVOS")
print("=" * 45)

for i in range(len(y_test)):

    if y_test[i] == 0 and y_pred[i] == 1:

        print(
            f"{rutas[i].name} | "
            f"POLY: {probabilidades[i, 0]:.4f} | "
            f"NORM: {probabilidades[i, 1]:.4f}"
        )

print("\nEvaluación finalizada.")
print(f"Resultados guardados en: {RESULTADOS}")


from pathlib import Path
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from keras.applications.vgg19 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

RUTA_HISTORIAL = BASE_DIR / "modelos" / "historial.json"
RUTA_MODELO = BASE_DIR / "modelos" / "vgg19_colon.keras"
RUTA_CC = BASE_DIR / "dataset" / "CC"

CARPETA_GRAFICAS = BASE_DIR / "resultados" / "graficas"
CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)

CLASES = ["POLY", "NORM"]
TAMANO = (150, 150)

sns.set_theme(style="whitegrid")



def guardar(nombre):

    ruta_png = CARPETA_GRAFICAS / nombre

    ruta_pdf = CARPETA_GRAFICAS / (
        Path(nombre).stem + ".pdf"
    )

    plt.tight_layout()

    # Guardar imagen de alta resolución
    plt.savefig(
        ruta_png,
        dpi=300,
        bbox_inches="tight"
    )

    # Guardar versión vectorial
    plt.savefig(
        ruta_pdf,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Gráfica PNG: {ruta_png.name}")
    print(f"Gráfica PDF: {ruta_pdf.name}")



# ==========================================
# 2. CARGAR HISTORIAL
# ==========================================

with open(RUTA_HISTORIAL, "r", encoding="utf-8") as f:
    historial = json.load(f)

epocas = range(1, len(historial["accuracy"]) + 1)


# ==========================================
# 3. GRÁFICA DE EXACTITUD MEJORADA
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    epocas,
    np.array(historial["accuracy"]) * 100,
    marker="o",
    linewidth=2,
    markersize=7,
    label="Entrenamiento"
)

plt.plot(
    epocas,
    np.array(historial["val_accuracy"]) * 100,
    marker="s",
    linewidth=2,
    markersize=7,
    label="Validación"
)

plt.title("Exactitud durante el entrenamiento - VGG19")
plt.xlabel("Épocas")
plt.ylabel("Exactitud (%)")

plt.xticks(list(epocas))

# Ampliar la escala para distinguir pequeñas diferencias
plt.ylim(95, 100.5)

plt.yticks(np.arange(95, 100.6, 0.5))

plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

guardar("01_accuracy.png")



# ==========================================
# 4. GRÁFICA DE LOSS
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    epocas,
    historial["loss"],
    marker="o",
    label="Entrenamiento"
)

plt.plot(
    epocas,
    historial["val_loss"],
    marker="s",
    label="Validación"
)

plt.title("Pérdida durante el entrenamiento - VGG19")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.xticks(list(epocas))
plt.legend()

guardar("02_loss.png")


# ==========================================
# 5. CARGAR DATOS DE EVALUACIÓN
# ==========================================

def cargar_cc():
    imagenes = []
    etiquetas = []

    for etiqueta, categoria in enumerate(CLASES):

        ruta = RUTA_CC / categoria
        archivos = sorted(ruta.glob("*.jpg"))

        print(f"Cargando {categoria}: {len(archivos)} imágenes")

        for archivo in archivos:
            imagen = cv2.imread(str(archivo))

            if imagen is None:
                raise ValueError(f"No se pudo leer {archivo}")

            imagen = cv2.cvtColor(
                imagen, cv2.COLOR_BGR2RGB
            )

            imagen = cv2.resize(imagen, TAMANO)

            imagenes.append(imagen)
            etiquetas.append(etiqueta)

    X = np.asarray(imagenes, dtype=np.float32)
    Y = np.asarray(etiquetas, dtype=np.int32)

    X = preprocess_input(X)

    return X, Y


print("\nCargando modelo y dataset CC...")

modelo = load_model(RUTA_MODELO)

x_test, y_test = cargar_cc()

probabilidades = modelo.predict(
    x_test,
    batch_size=8,
    verbose=1
)

y_pred = np.argmax(probabilidades, axis=1)


# ==========================================
# 6. MÉTRICAS POR CATEGORÍA MEJORADAS
# ==========================================

precision, recall, f1, soporte = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[0, 1],
        zero_division=0
    )
)

posiciones = np.arange(len(CLASES))
ancho = 0.24

fig, ax = plt.subplots(figsize=(10, 6))

barras_precision = ax.bar(
    posiciones - ancho,
    precision * 100,
    ancho,
    label="Precisión"
)

barras_recall = ax.bar(
    posiciones,
    recall * 100,
    ancho,
    label="Sensibilidad"
)

barras_f1 = ax.bar(
    posiciones + ancho,
    f1 * 100,
    ancho,
    label="F1-score"
)

# Agregar porcentaje sobre cada barra
for barras in [
    barras_precision,
    barras_recall,
    barras_f1
]:
    ax.bar_label(
        barras,
        fmt="%.2f%%",
        padding=4,
        fontsize=10
    )

ax.set_xticks(posiciones)
ax.set_xticklabels(CLASES)

ax.set_ylim(0, 112)

ax.set_ylabel("Porcentaje (%)")
ax.set_title(
    "Precisión, sensibilidad y F1-score - VGG19"
)

ax.legend(
    loc="upper right",
    ncol=3
)

ax.grid(axis="y", alpha=0.25)

guardar("03_metricas_clasificacion.png")



# ==========================================
# 7. GRÁFICA DE ACIERTOS Y ERRORES
# ==========================================

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

correctos = np.diag(matriz)
errores = matriz.sum(axis=1) - correctos

plt.figure(figsize=(9, 5))

plt.bar(
    CLASES,
    correctos,
    label="Correctas"
)

plt.bar(
    CLASES,
    errores,
    bottom=correctos,
    label="Incorrectas"
)

for i in range(len(CLASES)):
    plt.text(
        i,
        correctos[i] / 2,
        str(correctos[i]),
        ha="center",
        va="center"
    )

    if errores[i] > 0:
        plt.text(
            i,
            correctos[i] + errores[i] + 2,
            str(errores[i]) + " errores",
            ha="center"
        )

plt.title("Clasificaciones correctas e incorrectas")
plt.ylabel("Cantidad de imágenes")
plt.legend()

guardar("04_aciertos_errores.png")


# ==========================================
# 8. MATRIZ DE CONFUSIÓN
# ==========================================

import pandas as pd

# Calcular matriz de confusión
matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

# Crear tabla con nombres de categorías
tabla_confusion = pd.DataFrame(
    matriz,
    index=["Real POLY", "Real NORM"],
    columns=["Predicho POLY", "Predicho NORM"]
)

# Mostrar resultados en terminal
print("\nMATRIZ DE CONFUSIÓN")
print("=" * 45)
print(tabla_confusion)

# Crear gráfica
plt.figure(figsize=(8, 6))

sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASES,
    yticklabels=CLASES,
    annot_kws={"size": 16},
    linewidths=0.5
)

plt.title(
    "Matriz de confusión - VGG19",
    fontsize=15
)

plt.xlabel("Categoría predicha")
plt.ylabel("Categoría real")

# Guardar figura
guardar("08_matriz_confusion.png")

# Guardar tabla en CSV
RUTA_CSV = (
    CARPETA_GRAFICAS / "08_matriz_confusion.csv"
)

tabla_confusion.to_csv(
    RUTA_CSV,
    encoding="utf-8-sig"
)

print(f"Matriz guardada en: {RUTA_CSV}")

# ==========================================
# 8. DISTRIBUCIÓN DE PUNTUACIONES POLY
# ==========================================

puntuaciones_poly = probabilidades[:, 0]

plt.figure(figsize=(9, 5))

plt.hist(
    puntuaciones_poly[y_test == 0],
    bins=25,
    alpha=0.65,
    label="POLY real"
)

plt.hist(
    puntuaciones_poly[y_test == 1],
    bins=25,
    alpha=0.65,
    label="NORM real"
)

plt.axvline(
    x=0.5,
    linestyle="--",
    label="Umbral 0.5"
)

plt.title("Distribución de puntuaciones POLY")
plt.xlabel("Puntuación asignada a POLY")
plt.ylabel("Cantidad de imágenes")
plt.legend()

guardar("05_distribucion_puntuaciones.png")


# ==========================================
# DISTRIBUCIÓN AMPLIADA DE PUNTUACIONES
# ==========================================

# Rango de puntuaciones que queremos analizar
limite_inferior = 0.35
limite_superior = 0.65

# Seleccionar únicamente imágenes dentro del rango
mascara = (
    (puntuaciones_poly >= limite_inferior) &
    (puntuaciones_poly <= limite_superior)
)

# Separar imágenes reales por categoría
puntuaciones_poly_reales = puntuaciones_poly[
    mascara & (y_test == 0)
]

puntuaciones_norm_reales = puntuaciones_poly[
    mascara & (y_test == 1)
]

plt.figure(figsize=(11, 6))

bins = np.linspace(
    limite_inferior,
    limite_superior,
    31
)

plt.hist(
    puntuaciones_poly_reales,
    bins=bins,
    alpha=0.7,
    label="POLY real"
)

plt.hist(
    puntuaciones_norm_reales,
    bins=bins,
    alpha=0.7,
    label="NORM real"
)

# Umbral de clasificación
plt.axvline(
    x=0.5,
    linestyle="--",
    linewidth=2,
    color="red",
    label="Umbral de clasificación (0.5)"
)

# Identificar falsos negativos
indices_fn = np.where(
    (y_test == 0) & (y_pred == 1)
)[0]

for i, indice in enumerate(indices_fn, start=1):

    puntuacion = puntuaciones_poly[indice]

    plt.axvline(
        x=puntuacion,
        linestyle=":",
        linewidth=1.5,
        color="black",
        alpha=0.7
    )

    plt.annotate(
        f"FN {i}: {puntuacion:.4f}",
        xy=(puntuacion, 0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 12 + i * 17),
        textcoords="offset points",
        rotation=90,
        fontsize=9
    )

plt.title(
    "Análisis de puntuaciones cercanas al umbral - VGG19"
)

plt.xlabel("Puntuación asignada a POLY")
plt.ylabel("Cantidad de imágenes")

plt.xlim(
    limite_inferior,
    limite_superior
)

plt.legend(loc="upper right")
plt.grid(alpha=0.25)

guardar("09_puntuaciones_ampliadas.png")



# ==========================================
# 9. CURVA ROC
# ==========================================

# POLY es la clase positiva.
y_binario = (y_test == 0).astype(int)

fpr, tpr, umbrales_roc = roc_curve(
    y_binario,
    puntuaciones_poly
)

area_roc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"VGG19 (AUC = {area_roc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Referencia aleatoria"
)

plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Sensibilidad")
plt.title("Curva ROC - VGG19")
plt.legend()

guardar("06_curva_roc.png")


# ==========================================
# 10. CURVA PRECISION-RECALL
# ==========================================

precision_curva, recall_curva, umbrales_pr = (
    precision_recall_curve(
        y_binario,
        puntuaciones_poly
    )
)

ap = average_precision_score(
    y_binario,
    puntuaciones_poly
)

plt.figure(figsize=(7, 6))

plt.plot(
    recall_curva,
    precision_curva,
    label=f"VGG19 (AP = {ap:.4f})"
)

plt.xlabel("Sensibilidad (Recall)")
plt.ylabel("Precisión")
plt.title("Curva Precision-Recall - VGG19")
plt.legend()

guardar("07_curva_precision_recall.png")


# ==========================================
# 11. RESUMEN FINAL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nGRÁFICAS GENERADAS CORRECTAMENTE")
print("=" * 45)

print(f"Accuracy evaluación: {accuracy:.4f}")
print(f"AUC ROC: {area_roc:.4f}")
print(f"Average Precision: {ap:.4f}")

print(f"\nCarpeta de resultados:\n{CARPETA_GRAFICAS}")

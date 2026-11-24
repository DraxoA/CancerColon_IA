
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

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
# 1. CONFIGURACIÓN GENERAL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

RUTA_HISTORIAL = (
    BASE_DIR / "resultados" / "mobilenetv2" / "historial.json"
)

RUTA_MODELO = (
    BASE_DIR / "modelos" / "mobilenetv2_colon.keras"
)

RUTA_CC = BASE_DIR / "dataset" / "CC"

CARPETA_GRAFICAS = (
    BASE_DIR / "resultados" / "mobilenetv2" / "graficas"
)

CARPETA_GRAFICAS.mkdir(
    parents=True,
    exist_ok=True
)

CLASES = ["POLY", "NORM"]
TAMANO = (150, 150)

sns.set_theme(style="whitegrid")


# ==========================================
# 2. FUNCIÓN PARA GUARDAR GRÁFICAS
# ==========================================

def guardar(nombre):

    ruta_png = CARPETA_GRAFICAS / nombre

    ruta_pdf = CARPETA_GRAFICAS / (
        Path(nombre).stem + ".pdf"
    )

    plt.tight_layout()

    # Guardar en PNG
    plt.savefig(
        ruta_png,
        dpi=300,
        bbox_inches="tight"
    )

    # Guardar en PDF
    plt.savefig(
        ruta_pdf,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Gráfica PNG: {ruta_png.name}")
    print(f"Gráfica PDF: {ruta_pdf.name}")


# ==========================================
# 3. CARGAR HISTORIAL DEL ENTRENAMIENTO
# ==========================================

print("\nCargando historial MobileNetV2...")

with open(
    RUTA_HISTORIAL,
    "r",
    encoding="utf-8"
) as archivo:

    historial = json.load(archivo)

epocas = np.arange(
    1,
    len(historial["accuracy"]) + 1
)


# ==========================================
# 4. GRÁFICA DE EXACTITUD (ACCURACY)
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

plt.title(
    "Exactitud durante el entrenamiento - MobileNetV2"
)

plt.xlabel("Épocas")
plt.ylabel("Exactitud (%)")

plt.xticks(epocas)

plt.ylim(95, 100.5)
plt.yticks(np.arange(95, 100.6, 0.5))

plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

guardar("01_accuracy.png")


# ==========================================
# 5. GRÁFICA DE PÉRDIDA (LOSS)
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    epocas,
    historial["loss"],
    marker="o",
    linewidth=2,
    label="Entrenamiento"
)

plt.plot(
    epocas,
    historial["val_loss"],
    marker="s",
    linewidth=2,
    label="Validación"
)

plt.title(
    "Pérdida durante el entrenamiento - MobileNetV2"
)

plt.xlabel("Épocas")
plt.ylabel("Loss")

plt.xticks(epocas)

plt.legend()
plt.grid(True, alpha=0.3)

guardar("02_loss.png")


# ==========================================
# 6. CARGAR IMÁGENES DE CC
# ==========================================

def cargar_cc():

    imagenes = []
    etiquetas = []
    rutas = []

    for etiqueta, categoria in enumerate(CLASES):

        ruta = RUTA_CC / categoria

        archivos = sorted(
            ruta.glob("*.jpg")
        )

        print(
            f"Cargando CC/{categoria}: "
            f"{len(archivos)} imágenes"
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
                TAMANO
            )

            imagenes.append(imagen)
            etiquetas.append(etiqueta)
            rutas.append(archivo)

    X = np.asarray(
        imagenes,
        dtype=np.float32
    )

    Y = np.asarray(
        etiquetas,
        dtype=np.int32
    )

    # Preprocesamiento propio de MobileNetV2
    X = preprocess_input(X)

    return X, Y, rutas


# ==========================================
# 7. CARGAR MODELO Y REALIZAR PREDICCIONES
# ==========================================

print("\nCargando modelo MobileNetV2...")

modelo = load_model(RUTA_MODELO)

print("\nPreparando imágenes de evaluación...")

x_test, y_test, rutas_test = cargar_cc()

print("\nRealizando predicciones...")

probabilidades = modelo.predict(
    x_test,
    batch_size=8,
    verbose=1
)

y_pred = np.argmax(
    probabilidades,
    axis=1
)

puntuaciones_poly = probabilidades[:, 0]


# ==========================================
# 8. MÉTRICAS POR CATEGORÍA
# ==========================================

precision, recall, f1, soporte = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[0, 1],
        zero_division=0
    )
)

posiciones = np.arange(
    len(CLASES)
)

ancho = 0.24

fig, ax = plt.subplots(
    figsize=(10, 6)
)

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
    "Precisión, sensibilidad y F1-score - MobileNetV2"
)

ax.legend(
    loc="upper right",
    ncol=3
)

ax.grid(
    axis="y",
    alpha=0.25
)

guardar("03_metricas_clasificacion.png")


# ==========================================
# 9. CLASIFICACIONES CORRECTAS E INCORRECTAS
# ==========================================

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

correctos = np.diag(matriz)

errores = (
    matriz.sum(axis=1) - correctos
)

plt.figure(figsize=(10, 6))

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
            f"{errores[i]} errores",
            ha="center"
        )

plt.title(
    "Clasificaciones correctas e incorrectas - MobileNetV2"
)

plt.ylabel("Cantidad de imágenes")

plt.legend()

guardar("04_aciertos_errores.png")


# ==========================================
# 10. DISTRIBUCIÓN DE PUNTUACIONES POLY
# ==========================================

plt.figure(figsize=(10, 6))

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
    linewidth=2,
    label="Umbral 0.5"
)

plt.title(
    "Distribución de puntuaciones POLY - MobileNetV2"
)

plt.xlabel("Puntuación asignada a POLY")
plt.ylabel("Cantidad de imágenes")

plt.legend()

guardar("05_distribucion_puntuaciones.png")


# ==========================================
# 11. CURVA ROC
# ==========================================

# POLY representa la clase positiva
y_binario = (
    y_test == 0
).astype(int)

fpr, tpr, umbrales_roc = roc_curve(
    y_binario,
    puntuaciones_poly
)

area_roc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"MobileNetV2 (AUC = {area_roc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Referencia aleatoria"
)

plt.xlabel(
    "Tasa de falsos positivos"
)

plt.ylabel(
    "Sensibilidad"
)

plt.title(
    "Curva ROC - MobileNetV2"
)

plt.legend()

guardar("06_curva_roc.png")


# ==========================================
# 12. CURVA PRECISION-RECALL
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

plt.figure(figsize=(8, 6))

plt.plot(
    recall_curva,
    precision_curva,
    linewidth=2,
    label=f"MobileNetV2 (AP = {ap:.4f})"
)

plt.xlabel(
    "Sensibilidad (Recall)"
)

plt.ylabel(
    "Precisión"
)

plt.title(
    "Curva Precision-Recall - MobileNetV2"
)

plt.legend()

guardar("07_curva_precision_recall.png")


# ==========================================
# 13. MATRIZ DE CONFUSIÓN
# ==========================================

tabla_confusion = pd.DataFrame(
    matriz,
    index=[
        "Real POLY",
        "Real NORM"
    ],
    columns=[
        "Predicho POLY",
        "Predicho NORM"
    ]
)

print("\nMATRIZ DE CONFUSIÓN")
print("=" * 45)
print(tabla_confusion)

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
    "Matriz de confusión - MobileNetV2",
    fontsize=15
)

plt.xlabel(
    "Categoría predicha"
)

plt.ylabel(
    "Categoría real"
)

guardar("08_matriz_confusion.png")

# Guardar matriz en CSV
RUTA_CSV = (
    CARPETA_GRAFICAS /
    "08_matriz_confusion.csv"
)

tabla_confusion.to_csv(
    RUTA_CSV,
    encoding="utf-8-sig"
)

print(
    f"Matriz guardada en: {RUTA_CSV}"
)


# ==========================================
# 14. PUNTUACIONES CERCANAS AL UMBRAL
# ==========================================

limite_inferior = 0.35
limite_superior = 0.65

mascara = (
    (puntuaciones_poly >= limite_inferior)
    &
    (puntuaciones_poly <= limite_superior)
)

puntuaciones_poly_reales = puntuaciones_poly[
    mascara & (y_test == 0)
]

puntuaciones_norm_reales = puntuaciones_poly[
    mascara & (y_test == 1)
]

fig, ax = plt.subplots(
    figsize=(11, 6)
)

bins = np.linspace(
    limite_inferior,
    limite_superior,
    31
)

ax.hist(
    puntuaciones_poly_reales,
    bins=bins,
    alpha=0.65,
    label="POLY real"
)

ax.hist(
    puntuaciones_norm_reales,
    bins=bins,
    alpha=0.65,
    label="NORM real"
)

# Umbral de decisión
ax.axvline(
    x=0.5,
    linestyle="--",
    linewidth=2,
    color="red",
    label="Umbral 0.5"
)

# Identificar falsos negativos
indices_fn = np.where(
    (y_test == 0) &
    (y_pred == 1)
)[0]

# Seleccionar los falsos negativos
# que están dentro del intervalo
indices_fn_visibles = [
    indice
    for indice in indices_fn
    if (
        limite_inferior
        <= puntuaciones_poly[indice]
        <= limite_superior
    )
]

# Representar los falsos negativos
# mediante marcas en la parte inferior

for indice in indices_fn_visibles:

    puntuacion = puntuaciones_poly[indice]

    ax.plot(
        puntuacion,
        0,
        marker="|",
        markersize=17,
        markeredgewidth=2,
        color="#AD3333"
    )

ax.set_title(
    "Análisis de puntuaciones cercanas al umbral - MobileNetV2"
)

ax.set_xlabel(
    "Puntuación asignada a POLY"
)

ax.set_ylabel(
    "Cantidad de imágenes"
)

ax.set_xlim(
    limite_inferior,
    limite_superior
)

ax.yaxis.get_major_locator().set_params(
    integer=True
)

ax.grid(alpha=0.25)

ax.legend(loc="upper right")

# Informar cuántos FN aparecen
ax.text(
    0.02,
    0.96,
    (
        f"FN totales: {len(indices_fn)}\n"
        f"FN visibles en este rango: "
        f"{len(indices_fn_visibles)}"
    ),
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.85
    }
)

guardar("09_puntuaciones_ampliadas.png")


# ==========================================
# 15. RESUMEN FINAL
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

vp = int(matriz[0, 0])
fn = int(matriz[0, 1])
fp = int(matriz[1, 0])
vn = int(matriz[1, 1])

print("\nGRÁFICAS MOBILENETV2 GENERADAS")
print("=" * 45)

print(f"Imágenes evaluadas: {len(y_test)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Verdaderos positivos: {vp}")
print(f"Verdaderos negativos: {vn}")

print(f"Falsos positivos: {fp}")
print(f"Falsos negativos: {fn}")

print(f"AUC ROC: {area_roc:.4f}")
print(f"Average Precision: {ap:.4f}")

print(
    f"\nGráficas guardadas en:\n"
    f"{CARPETA_GRAFICAS}"
)

print(
    "\nProceso finalizado correctamente."
)


from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DESTINO = (
    BASE_DIR / "resultados" / "comparacion_modelos"
)

DESTINO.mkdir(parents=True, exist_ok=True)

# Datos registrados de los entrenamientos
EXPERIMENTOS = {
    "VGG19": {
        "historial": BASE_DIR / "modelos" / "historial.json",
        "matriz": BASE_DIR / "resultados" / "graficas" / "08_matriz_confusion.csv",
        "segundos": 430.68,
        "parametros": 24383810,
        "entrenables": 9062658,
        "color": "#AD3333"
    },

    "MobileNetV2": {
        "historial": BASE_DIR / "resultados" / "mobilenetv2" / "historial.json",
        "matriz": BASE_DIR / "resultados" / "mobilenetv2" / "graficas" / "08_matriz_confusion.csv",
        "segundos": 82.96,
        "parametros": 2586434,
        "entrenables": 328450,
        "color": "#347E9D"
    }
}

sns.set_theme(
    style="whitegrid",
    font_scale=1.0
)


# ==========================================
# 2. LEER HISTORIAL
# ==========================================

def leer_historial(ruta):

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe el historial: {ruta}"
        )

    with ruta.open(encoding="utf-8") as archivo:
        datos = json.load(archivo)

    claves = (
        "accuracy",
        "val_accuracy",
        "loss",
        "val_loss"
    )

    for clave in claves:
        if clave not in datos:
            raise ValueError(
                f"Falta '{clave}' en {ruta}"
            )

    longitudes = {
        len(datos[clave])
        for clave in claves
    }

    if len(longitudes) != 1:
        raise ValueError(
            f"Las series tienen diferente longitud: {ruta}"
        )

    return datos


# ==========================================
# 3. LEER MATRIZ DE CONFUSIÓN
# ==========================================

def leer_matriz(ruta):

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe la matriz CSV: {ruta}"
        )

    tabla = pd.read_csv(
        ruta,
        index_col=0
    )

    filas = [
        "Real POLY",
        "Real NORM"
    ]

    columnas = [
        "Predicho POLY",
        "Predicho NORM"
    ]

    try:
        matriz = tabla.loc[
            filas,
            columnas
        ].to_numpy(dtype=int)

    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Formato incorrecto de matriz: {ruta}"
        ) from error

    if matriz.shape != (2, 2):
        raise ValueError(
            f"La matriz debe tener tamaño 2x2: {ruta}"
        )

    if np.any(matriz < 0):
        raise ValueError(
            f"La matriz contiene valores negativos: {ruta}"
        )

    return matriz


# ==========================================
# 4. CALCULAR MÉTRICAS
# ==========================================

def calcular_metricas(matriz):

    # POLY es nuestra clase positiva

    vp = int(matriz[0, 0])
    fn = int(matriz[0, 1])

    fp = int(matriz[1, 0])
    vn = int(matriz[1, 1])

    total = vp + fn + fp + vn

    precision = (
        vp / (vp + fp)
        if (vp + fp) > 0 else 0
    )

    sensibilidad = (
        vp / (vp + fn)
        if (vp + fn) > 0 else 0
    )

    especificidad = (
        vn / (vn + fp)
        if (vn + fp) > 0 else 0
    )

    f1 = (
        2 * precision * sensibilidad
        / (precision + sensibilidad)
        if (precision + sensibilidad) > 0 else 0
    )

    accuracy = (
        (vp + vn) / total
        if total > 0 else 0
    )

    return {
        "accuracy": accuracy,
        "precision_poly": precision,
        "sensibilidad_poly": sensibilidad,
        "f1_poly": f1,
        "especificidad": especificidad,
        "vp": vp,
        "vn": vn,
        "fp": fp,
        "fn": fn,
        "total": total
    }


# ==========================================
# 5. FUNCIÓN PARA GUARDAR GRÁFICAS
# ==========================================

def guardar(nombre):

    plt.tight_layout()

    figura = plt.gcf()

    for extension in ("png", "pdf"):

        ruta = DESTINO / f"{nombre}.{extension}"

        figura.savefig(
            ruta,
            dpi=300,
            bbox_inches="tight"
        )

    plt.close(figura)

    print(f"Generada: {nombre}.png / .pdf")


# ==========================================
# 6. CARGAR RESULTADOS DE AMBOS MODELOS
# ==========================================

resultados = {}

for nombre, config in EXPERIMENTOS.items():

    historial = leer_historial(
        config["historial"]
    )

    matriz = leer_matriz(
        config["matriz"]
    )

    resultados[nombre] = {
        "historial": historial,
        "matriz": matriz,
        **calcular_metricas(matriz),
        "segundos": config["segundos"],
        "parametros": config["parametros"],
        "entrenables": config["entrenables"],
        "color": config["color"],
        "epocas": len(historial["accuracy"])
    }

# Comprobar que ambos modelos se evaluaron
# con la misma cantidad de imágenes

totales = {
    resultado["total"]
    for resultado in resultados.values()
}

if len(totales) != 1:
    raise ValueError(
        "Los modelos no tienen el mismo número "
        "de imágenes de evaluación."
    )

nombres = list(resultados)

colores = [
    resultados[nombre]["color"]
    for nombre in nombres
]

x = np.arange(len(nombres))



# ==========================================
# 7. GRÁFICA DE EXACTITUD MEJORADA
# ==========================================

fig, ax = plt.subplots(figsize=(12, 7))

# Guardar valores para determinar la escala
todos_los_valores = []

for nombre in nombres:

    r = resultados[nombre]

    epocas = np.arange(
        1,
        r["epocas"] + 1
    )

    accuracy_train = (
        np.array(r["historial"]["accuracy"]) * 100
    )

    accuracy_val = (
        np.array(r["historial"]["val_accuracy"]) * 100
    )

    todos_los_valores.extend(accuracy_train)
    todos_los_valores.extend(accuracy_val)

    # Exactitud de entrenamiento
    ax.plot(
        epocas,
        accuracy_train,
        color=r["color"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=f"{nombre} - Entrenamiento"
    )

    # Exactitud de validación
    ax.plot(
        epocas,
        accuracy_val,
        color=r["color"],
        marker="s",
        linestyle="--",
        linewidth=2.5,
        markersize=7,
        label=f"{nombre} - Validación"
    )

# ==========================================
# ESCALA VERTICAL AUTOMÁTICA
# ==========================================

valor_minimo = min(todos_los_valores)
valor_maximo = max(todos_los_valores)

# Margen para evitar cortar los puntos
margen = max(
    0.5,
    (valor_maximo - valor_minimo) * 0.10
)

limite_inferior = max(
    0,
    valor_minimo - margen
)

limite_superior = min(
    101,
    valor_maximo + margen
)

ax.set_ylim(
    limite_inferior,
    limite_superior
)

# ==========================================
# PRESENTACIÓN
# ==========================================

ax.set_title(
    "Comparación de exactitud - VGG19 vs MobileNetV2",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Época", fontsize=12)
ax.set_ylabel("Exactitud (%)", fontsize=12)

max_epocas = max(
    r["epocas"]
    for r in resultados.values()
)

ax.set_xticks(
    range(1, max_epocas + 1)
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.35
)

ax.legend(
    loc="lower right",
    fontsize=10,
    frameon=True
)

# Indicar que la escala está ampliada
ax.text(
    0.02,
    0.03,
    "Nota: eje vertical con escala ampliada.",
    transform=ax.transAxes,
    fontsize=9,
    color="gray"
)

guardar("01_comparacion_accuracy")


# ==========================================
# 8. GRÁFICA DE PÉRDIDA
# ==========================================

fig, ax = plt.subplots(figsize=(10, 6))

for nombre in nombres:

    r = resultados[nombre]

    epocas = np.arange(
        1,
        r["epocas"] + 1
    )

    ax.plot(
        epocas,
        r["historial"]["loss"],
        color=r["color"],
        marker="o",
        linewidth=2,
        label=f"{nombre} - Entrenamiento"
    )

    ax.plot(
        epocas,
        r["historial"]["val_loss"],
        color=r["color"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label=f"{nombre} - Validación"
    )

ax.set_xlabel("Época")
ax.set_ylabel("Pérdida (Loss)")

ax.set_title(
    "Comparación de pérdida - VGG19 vs MobileNetV2"
)

ax.set_xticks(
    range(1, max_epocas + 1)
)

ax.legend(fontsize=9)

guardar("02_comparacion_loss")


# ==========================================
# 9. COMPARACIÓN DE MÉTRICAS
# ==========================================

etiquetas = [
    "Exactitud",
    "Precisión POLY",
    "Sensibilidad POLY",
    "F1 POLY",
    "Especificidad"
]

claves = [
    "accuracy",
    "precision_poly",
    "sensibilidad_poly",
    "f1_poly",
    "especificidad"
]

fig, ax = plt.subplots(figsize=(11, 6))

posiciones = np.arange(len(claves))

ancho = 0.35

for i, nombre in enumerate(nombres):

    r = resultados[nombre]

    valores = [
        100 * r[clave]
        for clave in claves
    ]

    barras = ax.bar(
        posiciones + (i - 0.5) * ancho,
        valores,
        width=ancho,
        color=r["color"],
        label=nombre
    )

    ax.bar_label(
        barras,
        fmt="%.2f",
        padding=3,
        fontsize=8
    )

ax.set_xticks(
    posiciones,
    etiquetas
)

ax.set_ylim(0, 110)

ax.set_ylabel("Porcentaje (%)")

ax.set_title(
    "Comparación de métricas de clasificación"
)

ax.legend()

guardar("03_comparacion_metricas")


# ==========================================
# 10. COMPARACIÓN DE ERRORES
# ==========================================

fig, ax = plt.subplots(figsize=(8, 5))

for i, clave in enumerate(["fn", "fp"]):

    valores = [
        resultados[nombre][clave]
        for nombre in nombres
    ]

    barras = ax.bar(
        x + (i - 0.5) * 0.35,
        valores,
        width=0.35,
        label=(
            "Falsos negativos"
            if clave == "fn"
            else "Falsos positivos"
        )
    )

    ax.bar_label(
        barras,
        padding=3
    )

ax.set_xticks(x, nombres)

ax.set_ylabel("Cantidad de imágenes")

ax.set_title(
    "Comparación de errores de clasificación"
)

ax.legend()

guardar("04_comparacion_errores")


# ==========================================
# 11. COMPARACIÓN DE TIEMPOS
# ==========================================

fig, ax = plt.subplots(figsize=(8, 5))

tiempos = [
    resultados[nombre]["segundos"]
    for nombre in nombres
]

barras = ax.bar(
    nombres,
    tiempos,
    color=colores
)

ax.bar_label(
    barras,
    fmt="%.2f s",
    padding=4
)

ax.set_ylim(
    0,
    max(tiempos) * 1.15
)

ax.set_ylabel("Segundos")

ax.set_title(
    "Comparación de tiempos de entrenamiento"
)

guardar("05_comparacion_tiempos")


# ==========================================
# 12. COMPARACIÓN DE PARÁMETROS
# ==========================================

fig, ax = plt.subplots(figsize=(9, 5))

ancho = 0.35

for i, clave in enumerate([
    "parametros",
    "entrenables"
]):

    valores = [
        resultados[nombre][clave] / 1000000
        for nombre in nombres
    ]

    barras = ax.bar(
        x + (i - 0.5) * ancho,
        valores,
        ancho,
        label=(
            "Parámetros totales"
            if clave == "parametros"
            else "Parámetros entrenables"
        )
    )

    ax.bar_label(
        barras,
        fmt="%.2f M",
        padding=3
    )

ax.set_xticks(x, nombres)

ax.set_ylabel(
    "Millones de parámetros"
)

ax.set_title(
    "Comparación del tamaño de los modelos"
)

ax.legend()

guardar("06_comparacion_parametros")


# ==========================================
# 13. COMPARACIÓN DE MATRICES DE CONFUSIÓN
# ==========================================

fig, ejes = plt.subplots(
    1,
    2,
    figsize=(11, 5)
)

maximo = max(
    int(resultados[nombre]["matriz"].max())
    for nombre in nombres
)

for ax, nombre in zip(ejes, nombres):

    sns.heatmap(
        resultados[nombre]["matriz"],
        annot=True,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=maximo,
        xticklabels=["POLY", "NORM"],
        yticklabels=["POLY", "NORM"],
        cbar=False,
        ax=ax,
        annot_kws={"fontsize": 14}
    )

    ax.set_title(nombre)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Categoría real")

fig.suptitle(
    "Comparación de matrices de confusión",
    fontsize=14
)

guardar("07_matrices_confusion")


# ==========================================
# 14. CREAR TABLA COMPARATIVA
# ==========================================

filas = []

for nombre in nombres:

    r = resultados[nombre]

    filas.append({
        "Modelo": nombre,
        "Imágenes evaluadas": r["total"],
        "Accuracy (%)": round(
            r["accuracy"] * 100, 4
        ),
        "Precisión POLY (%)": round(
            r["precision_poly"] * 100, 4
        ),
        "Sensibilidad POLY (%)": round(
            r["sensibilidad_poly"] * 100, 4
        ),
        "F1 POLY (%)": round(
            r["f1_poly"] * 100, 4
        ),
        "Especificidad (%)": round(
            r["especificidad"] * 100, 4
        ),
        "VP": r["vp"],
        "VN": r["vn"],
        "FP": r["fp"],
        "FN": r["fn"],
        "Épocas": r["epocas"],
        "Tiempo entrenamiento (s)": r["segundos"],
        "Parámetros totales": r["parametros"],
        "Parámetros entrenables": r["entrenables"]
    })

tabla = pd.DataFrame(filas)

RUTA_TABLA = (
    DESTINO / "08_tabla_comparativa.csv"
)

tabla.to_csv(
    RUTA_TABLA,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 15. RESUMEN FINAL
# ==========================================

print("\nCOMPARACIÓN COMPLETADA")
print("=" * 55)

print(
    tabla.to_string(index=False)
)

print(
    f"\nArchivos guardados en:\n{DESTINO}"
)

print(
    "\nNota: comparación experimental."
    "\nLa independencia de las imágenes "
    "a nivel de paciente no está confirmada."
)

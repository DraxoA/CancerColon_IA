
from pathlib import Path

# Obtener la ubicación del proyecto
BASE_DIR = Path(__file__).resolve().parent

# Carpeta donde almacenamos las imágenes
DATASET = BASE_DIR / "dataset"

# Conjuntos y categorías
conjuntos = ["train", "CC"]
categorias = ["NORM", "POLY"]

print("VERIFICACIÓN DEL DATASET")
print("-" * 40)

for conjunto in conjuntos:
    total = 0

    for categoria in categorias:
        ruta = DATASET / conjunto / categoria

        if not ruta.exists():
            print(f"ERROR: No existe {ruta}")
            continue

        imagenes = list(ruta.glob("*.jpg"))
        cantidad = len(imagenes)

        print(f"{conjunto}/{categoria}: {cantidad} imágenes")
        total += cantidad

    print(f"TOTAL {conjunto}: {total} imágenes")
    print("-" * 40)

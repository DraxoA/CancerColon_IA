
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from keras.optimizers import Adam

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

TAMANO_ENTRADA = (150, 150, 3)
TASA_APRENDIZAJE = 0.0001

# ==========================================
# 2. CREAR MODELO MOBILENETV2
# ==========================================

def crear_modelo_mobilenet():

    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=TAMANO_ENTRADA
    )

    # Congelar la base preentrenada
    base.trainable = False

    # Entrada del modelo
    entrada = Input(shape=TAMANO_ENTRADA)

    # Extraer características con MobileNetV2
    x = base(entrada, training=False)

    # Reducir mapas de características
    x = GlobalAveragePooling2D()(x)

    # Clasificador personalizado
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)

    salida = Dense(
        2,
        activation="softmax"
    )(x)

    modelo = Model(
        inputs=entrada,
        outputs=salida
    )

    # ======================================
    # 3. COMPILAR
    # ======================================

    modelo.compile(
        optimizer=Adam(
            learning_rate=TASA_APRENDIZAJE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


# ==========================================
# 4. VERIFICAR ARQUITECTURA
# ==========================================

if __name__ == "__main__":

    modelo = crear_modelo_mobilenet()

    modelo.summary()

    print(
        "\nModelo MobileNetV2 creado correctamente."
    )


from keras.applications import VGG19
from keras.models import Model
from keras.layers import (
    Flatten,
    BatchNormalization,
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
# 2. CARGAR VGG19 PREENTRENADA
# ==========================================

def crear_modelo():

    base_model = VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=TAMANO_ENTRADA
    )

    # Congelar las primeras 19 capas,
    # siguiendo el código original.
    for capa in base_model.layers[:19]:
        capa.trainable = False

    # ======================================
    # 3. CAPAS DE CLASIFICACIÓN
    # ======================================

    x = base_model.output
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)

    salida = Dense(
        2,
        activation="softmax"
    )(x)

    modelo = Model(
        inputs=base_model.input,
        outputs=salida
    )

    # ======================================
    # 4. COMPILAR
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
# 5. PROBAR LA ARQUITECTURA
# ==========================================

if __name__ == "__main__":

    modelo = crear_modelo()

    modelo.summary()

    print("\nModelo VGG19 creado correctamente.")

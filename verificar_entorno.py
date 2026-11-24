import sys
import tensorflow as tf

print("Versión de Python:", sys.version)
print("Versión de TensorFlow:", tf.__version__)

print("CPU disponibles:")
print(tf.config.list_physical_devices("CPU"))

print("GPU disponibles:")
print(tf.config.list_physical_devices("GPU"))
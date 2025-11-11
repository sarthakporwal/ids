import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
import numpy as np


print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs Available: {len(gpus)}")
    for gpu in gpus:
        print(f"- {gpu}")
else:
    print("No GPUs found. TensorFlow will use the CPU.")


model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(100,)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

X_train = np.random.randn(100, 100).astype(np.float32)
y_train = np.random.randint(0, 10, size=(100,))

model.fit(X_train, y_train, epochs=10000, batch_size=50)

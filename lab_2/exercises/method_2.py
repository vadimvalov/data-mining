import tkinter as tk
import numpy as np
from utils.inputField import InputField
import tensorflow as tf
from tensorflow.keras import models, layers

def create_and_train_cnn():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (2, 2), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
    model.save("mnist_cnn.keras")
    return model

def load_or_train_cnn():
    try:
        return tf.keras.models.load_model("mnist_cnn.keras")
    except:
        return create_and_train_cnn()

model = load_or_train_cnn()

def recognize(img_array):
    img_reshaped = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_reshaped, verbose=0)
    return np.argmax(prediction)

root = tk.Tk()
root.title("CNN")
InputField(root, recognize)
root.mainloop()
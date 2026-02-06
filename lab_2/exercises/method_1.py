import tkinter as tk
import numpy as np
from utils.inputField import InputField
import tensorflow as tf
from tensorflow.keras import models, layers

def create_and_train_ann():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128),
        layers.Dense(40, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
    model.save("mnist_ann.keras")
    return model

def load_or_train_ann():
    try:
        return tf.keras.models.load_model("mnist_ann.keras")
    except:
        return create_and_train_ann()

model = load_or_train_ann()

def recognize(img_array):
    img_flat = img_array.reshape(1, 784)
    prediction = model.predict(img_flat, verbose=0)
    return np.argmax(prediction)

root = tk.Tk()
root.title("ANN")
InputField(root, recognize)
root.mainloop()
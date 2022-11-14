import keras
import keras.layers as layers
import tensorflow as tf
import pandas as pd

from Lib.read_dataset import read_dataset



def run():
    x_train, x_test, y_train, y_test = read_dataset(img_size=28, useLabels=["Male", "Smiling", "Young"])

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(y_train.shape[1], activation='softmax')
    ])

    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam,
                    loss="mse",
                    metrics=['accuracy'])

    # plot model
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    # train model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    

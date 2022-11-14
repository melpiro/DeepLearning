import keras
import keras.layers as layers
import tensorflow as tf
import pandas as pd





def run():
    dataset = pd.read_csv('./dataset/data.csv')

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(dataset['data'], dataset['label'], epochs=5, batch_size=32)

    

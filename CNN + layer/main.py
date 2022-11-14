import keras
import keras.layers as layers
import tensorflow as tf
import pandas as pd

from Lib.read_dataset import read_dataset
from Lib.benchmark import benchmarkit
from Lib.eval import evaluate_output


@benchmarkit
def run():
    labels = ["Male"]
    labels_opposite = ["Female"]
    img_size = 64
    x_train, x_test, y_train, y_test = read_dataset(img_size=img_size, useLabels=labels)

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(y_train.shape[1], activation='sigmoid')
    ])

    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam,
                    loss="binary_crossentropy",
                    metrics=['accuracy'])

    # plot model
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    # train model
    model.fit(x_train, y_train, epochs=2, batch_size=32, validation_data=(x_test, y_test))

    
    # plot some images
    import matplotlib.pyplot as plt
    n = 4
    fig, ax = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n):
        for j in range(n):
            ax[i, j].imshow(x_test[i*n+j])
            ax[i, j].axis('off')



            preds = model.predict(x_test[i*n+j].reshape(1, img_size, img_size, 3))
            true = y_test[i*n+j]
            evaluate_output(labels, true, preds)

            title = ''
            t = []
            for k in range(len(labels)):
                if true[k] > 0.5:
                    t.append(labels[k])
                else:
                    t.append(labels_opposite[k])
            title = ', '.join(t)

            t = []
            for k in range(len(labels)):
                if preds[0][k] > 0.5:
                    t.append(labels[k])
                else:
                    t.append(labels_opposite[k])
            title += '\n' + ', '.join(t)

            ax[i, j].set_title(title)

    fig.tight_layout()
    plt.show()
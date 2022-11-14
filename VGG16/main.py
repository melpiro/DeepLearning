import keras
import keras.layers as layers
import tensorflow as tf
import pandas as pd

from Lib.read_dataset import read_dataset



def run():


    labels = ["Male"]
    labels_opposite = ["Femal"]
    img_size = 75
    x_train, x_test, y_train, y_test = read_dataset(img_size=img_size, useLabels=labels)

    base_model = tf.keras.applications.VGG16(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(img_size, img_size, 3),
        include_top=False
    )

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)


    model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    # train model
    model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))

    base_model.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
    model.fit(x_train, y_train, epochs=6, batch_size=32, validation_data=(x_test, y_test))


    
    # plot some images
    import matplotlib.pyplot as plt
    n = 4
    fig, ax = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n):
        for j in range(n):
            ax[i, j].imshow(x_test[i*n+j])
            ax[i, j].axis('off')



            preds = model.predict(x_train)
            true = y_test[i*n+j]


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

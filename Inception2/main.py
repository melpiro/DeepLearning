
import keras
import keras.layers as layers
import tensorflow as tf
import pandas as pd

from Lib.read_dataset import read_dataset


from keras.models import Model
from keras.layers import Conv2D , MaxPool2D , Input , GlobalAveragePooling2D ,AveragePooling2D, Dense , Dropout ,Activation, Flatten , BatchNormalization


def conv_with_Batch_Normalisation(prev_layer , nbr_kernels , filter_Size , strides =(1,1) , padding = 'same'):
    x = Conv2D(filters=nbr_kernels, kernel_size = filter_Size, strides=strides , padding=padding)(prev_layer)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    return x

def StemBlock(prev_layer):
    x = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 32, filter_Size=(3,3) , strides=(2,2))
    x = conv_with_Batch_Normalisation(x, nbr_kernels = 32, filter_Size=(3,3))
    x = conv_with_Batch_Normalisation(x, nbr_kernels = 64, filter_Size=(3,3))
    x = MaxPool2D(pool_size=(3,3) , strides=(2,2)) (x)
    x = conv_with_Batch_Normalisation(x, nbr_kernels = 80, filter_Size=(1,1))
    x = conv_with_Batch_Normalisation(x, nbr_kernels = 192, filter_Size=(3,3))
    x = MaxPool2D(pool_size=(3,3) , strides=(2,2)) (x)
    
    return x    
    

def InceptionBlock_A(prev_layer  , nbr_kernels):
    
    branch1 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 32, filter_Size = (1,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels=64, filter_Size=(3,3))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels=64, filter_Size=(3,3))
    
    branch2 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels=32, filter_Size=(1,1))
    branch2 = conv_with_Batch_Normalisation(branch2, nbr_kernels=64, filter_Size=(3,3)) # may be 3*3
    
    branch3 = AveragePooling2D(pool_size=(3,3) , strides=(1,1) , padding='same') (prev_layer)
    branch3 = conv_with_Batch_Normalisation(branch3, nbr_kernels = nbr_kernels, filter_Size = (1,1))
    
    branch4 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels=32, filter_Size=(1,1))
    
    output = layers.Concatenate()([branch1 , branch2 , branch3 , branch4])
    
    return output



def InceptionBlock_B(prev_layer , nbr_kernels):
    
    branch1 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = nbr_kernels, filter_Size = (1,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = nbr_kernels, filter_Size = (7,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = nbr_kernels, filter_Size = (1,7))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = nbr_kernels, filter_Size = (7,1))    
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 192, filter_Size = (1,7))
    
    branch2 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = nbr_kernels, filter_Size = (1,1))
    branch2 = conv_with_Batch_Normalisation(branch2, nbr_kernels = nbr_kernels, filter_Size = (1,7))
    branch2 = conv_with_Batch_Normalisation(branch2, nbr_kernels = 192, filter_Size = (7,1))
    
    branch3 = AveragePooling2D(pool_size=(3,3) , strides=(1,1) , padding ='same') (prev_layer)
    branch3 = conv_with_Batch_Normalisation(branch3, nbr_kernels = 192, filter_Size = (1,1))
    
    branch4 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 192, filter_Size = (1,1))
    
    output = layers.Concatenate()([branch1 , branch2 , branch3 , branch4])
    
    return output    


    
def InceptionBlock_C(prev_layer):
    
    branch1 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 448, filter_Size = (1,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 384, filter_Size = (3,3))
    branch1_1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 384, filter_Size = (1,3))    
    branch1_2 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 384, filter_Size = (3,1))
    branch1 = layers.Concatenate()([branch1_1 , branch1_2])
    
    branch2 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 384, filter_Size = (1,1))
    branch2_1 = conv_with_Batch_Normalisation(branch2, nbr_kernels = 384, filter_Size = (1,3))
    branch2_2 = conv_with_Batch_Normalisation(branch2, nbr_kernels = 384, filter_Size = (3,1))
    branch2 = layers.Concatenate()([branch2_1 , branch2_2])
    
    branch3 = AveragePooling2D(pool_size=(3,3) , strides=(1,1) , padding='same')(prev_layer)
    branch3 = conv_with_Batch_Normalisation(branch3, nbr_kernels = 192, filter_Size = (1,1))
    
    branch4 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 320, filter_Size = (1,1))
    
    output = layers.Concatenate()([branch1 , branch2 , branch3 , branch4])
    
    return output

def ReductionBlock_A(prev_layer):
    
    branch1 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 32, filter_Size = (1,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 64, filter_Size = (3,3))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 64, filter_Size = (3,3) , strides=(2,2) ) #, padding='valid'
    
    branch2 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 64, filter_Size=(3,3) , strides=(2,2) )
    
    branch3 = MaxPool2D(pool_size=(3,3) , strides=(2,2) , padding='same')(prev_layer)
    
    output = layers.Concatenate()([branch1 , branch2 , branch3])
    
    return output

    

def ReductionBlock_B(prev_layer):
    
    branch1 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 192, filter_Size = (1,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 192, filter_Size = (1,7))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 192, filter_Size = (7,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 192, filter_Size = (3,3) , strides=(2,2) , padding = 'valid')
    
    branch2 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 192, filter_Size = (1,1) )
    branch2 = conv_with_Batch_Normalisation(branch2, nbr_kernels = 320, filter_Size = (3,3) , strides=(2,2) , padding='valid' )

    branch3 = MaxPool2D(pool_size=(3,3) , strides=(2,2) )(prev_layer)
    
    output = layers.Concatenate()([branch1 , branch2 , branch3])
    
    return output

# def auxiliary_classifier(prev_Layer):
#     x = AveragePooling2D(pool_size=(5,5) , strides=(3,3)) (prev_Layer)
#     x = conv_with_Batch_Normalisation(x, nbr_kernels = 128, filter_Size = (1,1))
#     x = Flatten()(x)
#     x = Dense(units = 768, activation='relu') (x)
#     x = Dropout(rate = 0.2) (x)
#     x = Dense(units = 1, activation='sigmoid') (x)
#     return x



def InceptionV3(img_size, output_size):
    
    input_layer = Input(shape=(img_size , img_size , 3))
    
    x = InceptionBlock_A(prev_layer = input_layer ,nbr_kernels = 32)
    x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same')(x)
    x = InceptionBlock_A(prev_layer = x ,nbr_kernels = 32)
    x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same')(x)

    x = InceptionBlock_A(prev_layer = x ,nbr_kernels = 32)

    x= Conv2D(filters = 32, kernel_size = (1,1) , strides=(1,1) , padding='same')(x)
    
    x = Flatten()(x)
    x = Dense(units=2048, activation='relu') (x)
    x = Dropout(rate = 0.2) (x)
    x = Dense(units=output_size, activation='sigmoid') (x)
    
    model = Model(inputs = input_layer , outputs = x , name = 'Inception-V3')
    
    return model


def run():


    labels = ["Male"]
    labels_opposite = ["Femal"]
    img_size = 32
    x_train, x_test, y_train, y_test = read_dataset(img_size=img_size, useLabels=labels)

    model = InceptionV3(img_size, len(labels))

    model.compile(optimizer=keras.optimizers.Adam(lr = 0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    # train model
    model.fit(x_train, y_train, epochs=6, batch_size=64, validation_data=(x_test, y_test))

    
    # plot some images
    import matplotlib.pyplot as plt
    n = 4
    fig, ax = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n):
        for j in range(n):
            ax[i, j].imshow(x_test[i*n+j])
            ax[i, j].axis('off')



            preds = model.predict(x_train[i*n+j].reshape(1, img_size, img_size, 3))
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

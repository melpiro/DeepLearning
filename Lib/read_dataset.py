
import numpy as np
import pandas as pd
from keras.utils import load_img, img_to_array


def flipImg(img):
    return np.fliplr(img)

def rotate90(img):
    return np.rot90(img)

def rotate180(img):
    return np.rot90(img, k=2)

def rotate270(img):
    return np.rot90(img, k=3)
    


def read_dataset(img_size= 64, useLabels = ["Male"]):


    dataset = pd.read_csv('./Dataset/list_attr_celeba_small.csv')
    dataset = dataset[:11000]


    images = np.zeros((len(dataset), img_size, img_size, 3), dtype=np.float32)
    labels = np.zeros((len(dataset), len(useLabels)))

    for i in range(len(dataset)):
        path = dataset['image_id'][i]
        img = load_img('./Dataset/img_align_celeba_small/' + path, target_size=(img_size, img_size))
        img = img_to_array(img)/255.0

        n = np.random.randint(0, 5)
        if n == 1:
            img = flipImg(img)
        elif n == 2:
            img = rotate90(img)
        elif n == 3:
            img = rotate180(img)
        elif n == 4:
            img = rotate270(img)
            

        images[i] = img
        labels[i] = dataset.loc[i, useLabels]

    labels = (labels + 1.0) / 2.0

    image_train = images[:10000]
    image_test = images[10000:]
    label_train = labels[:10000]
    label_test = labels[10000:]

    return image_train, image_test, label_train, label_test



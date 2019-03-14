import os
from sklearn.utils import shuffle
import skimage
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
from skimage.color import rgb2hsv
from sklearn import metrics, model_selection
import tensorflow as tf
from tensorflow.contrib import learn
import scipy

from skimage import data, io, restoration,segmentation, filters
from skimage.color import rgb2gray
from scipy.signal import convolve2d


def eliminacionRuido(imagen , mostrar=False):
    grayscale = rgb2gray(imagen)
    rst_DNM =filters.gaussian(grayscale,sigma=1.5)
    if mostrar:
        fig, axes = plt.subplots(1, 3, figsize=(8, 4))
        ax = axes.ravel()
        ax[0].imshow(imagen)
        ax[0].set_title("Original")
        ax[1].imshow(rst_DNM, cmap=plt.cm.gray)
        ax[1].set_title("Desnoise")
        ax[2].imshow(grayscale, cmap=plt.cm.gray)
        ax[2].set_title("GrayScale")
        fig.tight_layout()
        plt.show()
    return rst_DNM

def segmentacion(imagen, mostrar = False):
    """labelArray = measure.label(pixel_array_numpy, return_num=True, neighbors=4)
    print(labelArray)

    imagenSegmentada = segmentation.quickshift(image,convert2lab=False)
    io.imsave('segmetada1.jpg', imagenSegmentada)"""
    """imagenSegmentada=segmentation.random_walker(imagen, labelArray)"""
    img =imagen
    thresh = filters.threshold_otsu(img)
    binary = img <= thresh
    if(mostrar):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        ax = axes.ravel()
        ax[0].imshow(imagen, cmap=plt.cm.gray)
        ax[0].set_title("Original")
        ax[1].imshow(binary, cmap=plt.cm.gray)
        ax[1].set_title("Segmentation")
        fig.tight_layout()
        plt.show()
    return binary

def leerMat(direccion, mostrar = False):
    matA = h5.File(direccion,'r')
    imagen =matA['/cjdata/image']
    if np.array(imagen).shape[0] != 512 or np.array(imagen).shape[1] != 512:
        return [],[]
    label = int(matA['/cjdata/label'][0][0])
    array = np.mat(imagen)
    imagenfloat= skimage.img_as_float(array)
    if mostrar:
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        ax = axes.ravel()
        ax[0].imshow(imagenfloat)
        ax[0].set_title("Original")
        ax[1].imshow(imagenfloat, cmap=plt.cm.magma)
        ax[1].set_title("Magama")
        ax[2].imshow(imagenfloat, cmap=plt.cm.gray)
        ax[2].set_title("Gray")
        fig.tight_layout()
        plt.show()

    return imagenfloat, label

def creacionDataset(num=3064):
    image_train = []
    label_train = []
    for i in range (1, num):
        imagen = leerMat("data/" + str(i) + ".mat")[0];
        if imagen != []:
            image_train.append(segmentacion(eliminacionRuido(imagen)))
            label_train.append(int(leerMat("data/" + str(i) + ".mat")[1])- 1)

    return image_train,label_train

def clasificacion():
    data =creacionDataset()
    print("soy una gueva")
    #use scikit.learn.datasets in the future
    print(len(data[0]),"gonorrea",len(data[1]))
    image_train = np.array(data[0])
    label_train = np.array(data[1])
    image_train =image_train.reshape(image_train.shape[0], image_train.shape[1] * image_train.shape[2])
    label_train = label_train.reshape(label_train.shape[0], )
    image_train, label_train = shuffle(image_train, label_train, random_state=42)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(image_train, label_train, test_size = .3, random_state = 42)
    #build 3 layer DNN with 10 20 10 units respectively
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
    classifier = learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10,20,10],n_classes=3)
    # #fit and predict
    classifier.fit(x_train, y_train, steps = 200)
    x_predict = classifier.predict_classes(x_test)
    x_predict = [x for x in x_predict ]
    score = metrics.accuracy_score(y_test, x_predict)
    print('Accuracy: {0:f}'.format(score))

if __name__ == "__main__":
    clasificacion()

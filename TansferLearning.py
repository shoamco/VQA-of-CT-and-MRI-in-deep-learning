

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow
import numpy as np
import pydot
from matplotlib import pyplot
import pandas as pd
from keras import applications
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

from keras.applications.vgg19 import VGG19
# from keras.applications.densenet.DenseNet169 import DenseNet169
from keras_applications import densenet

from sklearn.decomposition import PCA

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from urllib.request import urlretrieve
from os.path import isfile, isdir, getsize
from os import mkdir, makedirs, remove, listdir
from tqdm import tqdm

import zipfile
import tarfile
import pickle

from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator

import glob
import shutil

import pickle


#
def open_data():
    df = pd.read_csv('InputFiles/dataset.csv', names=['Images', 'Questions', 'Answers'])  # open csv file and rename columns
    # predictions = pd.read_csv('InputFiles/VQAM.csv', names=['Images', 'Questions', 'Answers'])  # open csv file and rename columns


    # # dictionary of replaceable words
    replace_dict = {"magnetic resonance imaging": "mri",
                    "mri scan": 'mri',
                    "MRI": "mri",
                    "shows": "show",
                    "reveal": "show",
                    "demonstrate": "show",
                    "CT": "ct",
                    "ct scan": "ct",
                    "does": "", "do ": "", "the": "",
                    # " a ":' ',' is ':' ',
                    }
    df.replace(to_replace=replace_dict, inplace=True, regex=True)  # replace word


    ImagesOfMri = df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') & df[
        'Answers'].str.contains('mri')) == True]['Images']
    ImagesOfCt = df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') & df[
        'Answers'].str.contains('ct')) == True]['Images']

    ImagesMri = ["images\Train-images\\" + img + ".jpg" for img in ImagesOfMri]
    ImagesCt = ["images\Train-images\\" + img + ".jpg" for img in ImagesOfCt]



    sizeMri=len(ImagesMri)
    sizeCt=len(ImagesCt)
    TrainData = np.concatenate((ImagesMri[:int(sizeMri*0.8)], ImagesCt[:int(sizeCt*0.8)]), axis=0, out=None)

    # TrainData = np.concatenate((ImagesMri[:int(sizeMri*0.8)], ImagesCt[:int(sizeCt*0.8)]), axis=0, out=None)
    y = np.full((len(ImagesMri[:int(sizeMri*0.8)])), 0)
    y1 = np.full((len(ImagesCt[:int(sizeCt*0.8)])), 1)
    TrainLabel= np.concatenate((y, y1), axis=0, out=None)

    validData= np.concatenate((ImagesMri[int(sizeMri*0.8):], ImagesCt[int(sizeCt*0.8):]), axis=0, out=None)
    y = np.full((len(ImagesMri[int(sizeMri*0.8):])), 0)
    y1 = np.full((len(ImagesCt[int(sizeCt*0.8):])), 1)
    validLabel=np.concatenate((y, y1), axis=0, out=None)
    return (TrainData,TrainLabel,validData,validLabel)

# Divide images into folders of train valid test
# In which each folder has a partition of CT MRI
# (To be able to use the model of keras)


def Divide_images_into_folders():
    TrainData, TrainLabel, validData, validLabel=open_data()
    makedirs('data')
    train_folder = 'data/train'
    valid_folder="data/valid"
    test_folder = 'data/test'

    if isdir(train_folder):  # if directory already exists
        shutil.rmtree(train_folder)
    if isdir(test_folder):  # if directory already exists
        shutil.rmtree(valid_folder)
    makedirs(train_folder + '/ct/')
    makedirs(train_folder + '/mri/')
    makedirs(valid_folder + '/ct/')
    makedirs(valid_folder + '/mri/')



    for f, i in zip(TrainData, TrainLabel):
        if i == 0:
            shutil.copy2(f, train_folder + '/ct/')
        else:
            shutil.copy2(f, train_folder + '/mri/')

    for f, i in zip(validData, validLabel):
        if i == 0:
            shutil.copy2(f, valid_folder + '/ct/')
        else:
            shutil.copy2(f, valid_folder + '/mri/')


if not os.path.exists('data'):
  Divide_images_into_folders()
TrainData, TrainLabel, validData, validLabel=open_data()


train_folder = 'data/train'
valid_folder="data/valid"
test_folder = 'data/test'



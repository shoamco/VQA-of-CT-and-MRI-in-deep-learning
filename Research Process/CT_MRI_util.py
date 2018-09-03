


import numpy as np
import matplotlib.pyplot as plt

from urllib.request import urlretrieve
from os.path import isfile, isdir, getsize
from os import mkdir, makedirs, remove
from tqdm import tqdm

import zipfile
import pickle

from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator

import glob
import shutil

import pickle

# %matplotlib inline

"""Download and extract the doge and cate pictures."""

catdog_dataset_folder_path = 'catdog'
train_folder = catdog_dataset_folder_path + '/train'
test_folder = catdog_dataset_folder_path + '/test'


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

# ---------------------------------------------------------------------
# download the files
# ---------------------------------------------------------------------
if not isfile('catdog.zip'):
    print('Download catdog.zip from download.microsoft.com...')
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Doge n Cate Dataset') as pbar:
        urlretrieve(
            'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip',
            'catdog.zip',
            pbar.hook)

# ---------------------------------------------------------------------
# Create the dataset folder
# ---------------------------------------------------------------------
if not isdir(catdog_dataset_folder_path):
    print('randomly choose 20000 images and moves them to training and the rest to testing folders...')

    mkdir(catdog_dataset_folder_path)
    with zipfile.ZipFile('catdog.zip') as f:
        f.extractall('./' + catdog_dataset_folder_path)
        # Unfortunately some of the files are corrupt so we need to clean these out:
        # !apt-get install -y jhead > /dev/null 2>&1
        # !jhead -de catdog/PetImages/Cat/*.jpg > /dev/null 2>&1
        # !jhead -de catdog/PetImages/Dog/*.jpg > /dev/null 2>&1

    files = glob.glob(catdog_dataset_folder_path + '/PetImages/**/*.jpg')
    labels = np.array([0] * 12500 + [1] * 12500)

    size = np.zeros(len(files))
    for i, f in enumerate(files):
        size[i] = getsize(f)

    idx = np.where(size == 0)[0]
    for i in idx[::-1]:
        del files[i]
        labels = np.delete(labels, i)

    """
    In keras it is required to place the training images in a certain folder, with the subfolders structured so that each subfolder contains the class. We will structure the validation folder in the same way:
    
    ```
    data/
        train/
            dogs/
                dog001.jpg
                dog002.jpg
                ...
            cats/
                cat001.jpg
                cat002.jpg
                ...
        validation/
            dogs/
                dog001.jpg
                dog002.jpg
                ...
            cats/
                cat001.jpg
                cat002.jpg
                ...
    ```            
    
    From the dataset we randomly choose 20000 images and moves them to training and the rest to testing folders.
    """

    len_data = len(files)
    train_examples = 20000
    test_examples = len_data - train_examples

    # randomly choose 20000 as training and testing cases
    permutation = np.random.permutation(len_data)
    train_set = [files[i] for i in permutation[:][:train_examples]]
    test_set = [files[i] for i in permutation[-test_examples:]]
    train_labels = labels[permutation[:train_examples]]
    test_labels = labels[permutation[-test_examples:]]


    if isdir(train_folder):  # if directory already exists
        shutil.rmtree(train_folder)
    if isdir(test_folder):  # if directory already exists
        shutil.rmtree(test_folder)
    makedirs(train_folder + '/cat/')
    makedirs(train_folder + '/dog/')
    makedirs(test_folder + '/cat/')
    makedirs(test_folder + '/dog/')

    for f, i in zip(train_set, train_labels):
        if i == 0:
            shutil.copy2(f, train_folder + '/cat/')
        else:
            shutil.copy2(f, train_folder + '/dog/')

    for f, i in zip(test_set, test_labels):
        if i == 0:
            shutil.copy2(f, test_folder + '/cat/')
        else:
            shutil.copy2(f, test_folder + '/dog/')



import pandas as pd


import os
import numpy as np
from os.path import  isdir
from os import  makedirs
import shutil
from pandas import ExcelWriter

def open_data():
    df = pd.read_csv('InputFiles/dataset.csv', names=['Images', 'Questions', 'Answers'])  # open csv file and rename columns


     # dictionary of replaceable words
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

    Mri = df[(~df['Questions'].str.contains(' mri | ct ') & df['Questions'].str.contains('what show') & df[
        'Answers'].str.contains(' mri ')) == True]
    Ct = df[(~df['Questions'].str.contains(' mri | ct ') & df['Questions'].str.contains('what show') & df[
        'Answers'].str.contains(' ct ')) == True]

    ImagesOfMri = df[(~df['Questions'].str.contains(' mri | ct ') & df['Questions'].str.contains('what show') & df[
        'Answers'].str.contains(' mri ')) == True]['Images']
    ImagesOfCt = df[(~df['Questions'].str.contains(' mri | ct ') & df['Questions'].str.contains('what show') & df[
        'Answers'].str.contains(' ct ')) == True]['Images']

    ImagesMri = ["images\Train-images\\" + img + ".jpg" for img in ImagesOfMri]
    ImagesCt = ["images\Train-images\\" + img + ".jpg" for img in ImagesOfCt]

    writer = ExcelWriter('outputFiles/MRI CT Answers.xlsx')

    Mri.to_excel(writer, 'ImagesOfMri', index=False)
    Ct.to_excel(writer, 'ImagesOfCt', index=False)

    writer.save()

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

    SizeTrain=len(TrainData)
    test_examples=len(validData)
    return (TrainData,TrainLabel,validData,validLabel,SizeTrain,test_examples)

# Divide images into folders of train valid test
# In which each folder has a partition of CT MRI
# (To be able to use the model of keras)


def Divide_images_into_folders():
    TrainData, TrainLabel, validData, validLabel,SizeTrain,test_examples=open_data()
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
            shutil.copy2(f, train_folder + '/mri/')
        else:
            shutil.copy2(f, train_folder + '/ct/')

    for f, i in zip(validData, validLabel):
        if i == 0:
            shutil.copy2(f, valid_folder + '/mri/')
        else:
            shutil.copy2(f, valid_folder + '/ct/')

# Data pre-processing and data augmentation
if not os.path.exists('data'):
  Divide_images_into_folders()
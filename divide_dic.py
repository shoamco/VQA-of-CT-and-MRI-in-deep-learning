

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

    Mri = df[(df['Questions'].str.contains(' mri  ') |  df['Answers'].str.contains(' mri ')) == True]
    Ct = df[(df['Questions'].str.contains(' ct ') | df['Answers'].str.contains(' ct ')) == True]



    ImagesMri = ["images\Train-images\\" + img + ".jpg" for img in Mri['Images']]
    ImagesCt = ["images\Train-images\\" + img + ".jpg" for img in Ct['Images']]

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

    validData= np.concatenate((ImagesMri[int(sizeMri*0.8):int(sizeMri*0.9)], ImagesCt[int(sizeCt*0.8):int(sizeCt*0.9)]), axis=0, out=None)
    y = np.full((len(ImagesMri[int(sizeMri*0.8):int(sizeMri*0.9)])), 0)
    y1 = np.full((len(ImagesCt[int(sizeCt*0.8):int(sizeCt*0.9)])), 1)
    validLabel=np.concatenate((y, y1), axis=0, out=None)
    
    
    testData= np.concatenate((ImagesMri[int(sizeMri*0.9):], ImagesCt[int(sizeCt*0.9):]), axis=0, out=None)
    y = np.full((len(ImagesMri[int(sizeMri*0.9):])), 0)
    y1 = np.full((len(ImagesCt[int(sizeCt*0.9):])), 1)
    testLabel=np.concatenate((y, y1), axis=0, out=None)

    SizeTrain=len(TrainData)
    test_examples=len(validData)
    SizeTest = len(testData)
    return (TrainData,TrainLabel,validData,validLabel,SizeTrain,test_examples,SizeTest,testData,testLabel)

# Divide images into folders of train valid test
# In which each folder has a partition of CT MRI
# (To be able to use the model of keras)


def Divide_images_into_folders():
    TrainData, TrainLabel, validData, validLabel,SizeTrain,test_examples,SizeTest,testData,testLabel=open_data()
    makedirs('data')
    train_folder = 'data/train'
    valid_folder="data/valid"
    test_folder = 'data/test'

    if isdir(train_folder):  # if directory already exists
        shutil.rmtree(train_folder)
    if isdir(valid_folder):  # if directory already exists
        shutil.rmtree(valid_folder)
    if isdir(test_folder):  # if directory already exists
            shutil.rmtree(test_folder)
    makedirs(train_folder + '/ct/')
    makedirs(train_folder + '/mri/')
    makedirs(valid_folder + '/ct/')
    makedirs(valid_folder + '/mri/')
    makedirs(test_folder + '/ct/')
    makedirs(test_folder + '/mri/')



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
    for f, i in zip(testData,testLabel):
        if i == 0:
            shutil.copy2(f, test_folder + '/mri/')
        else:
            shutil.copy2(f, test_folder + '/ct/')
# Data pre-processing and data augmentation
if not os.path.exists('data'):
  Divide_images_into_folders()
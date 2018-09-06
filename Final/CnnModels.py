

"""
This module contains the CNN model
Training the model on the training-set ,And geting prediction for the test-set

"""


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras.applications import Xception

from keras.applications import VGG16
from keras.applications.inception_v3 import InceptionV3
from sklearn.ensemble import RandomForestClassifier


import csv
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import glob
img_width, img_height = 256, 256



dataset_folder_path = 'data'
train_folder = dataset_folder_path + '/train'
valid_folder = dataset_folder_path + '/valid'
test_folder = dataset_folder_path + '/test'

train_files = glob.glob(train_folder + '/**/*.jpg')
valid_files = glob.glob(valid_folder + '/**/*.jpg')
test_files = glob.glob(test_folder + '/**/*.jpg')



nb_train_samples = len(train_files)
nb_validation_samples = len(valid_files)
nb_test_samples=len(test_files)

print("Number of train examples: " , nb_train_samples)
print("Number of test examples: ", nb_validation_samples)
print("Number of test examples: ", nb_test_samples)
channels = 3



"""
The function get: model ,epochs,batch_size and number of layers for freezing
Adds layers to classify the model - in the output layer
Training the model on the training-set
And returns prediction for the test-set
"""
def Train_Model_And_Predition(model,epochs,batch_size,FreezeLayers):
    # Freeze the layers which you don't want to train
    for layer in model.layers[:-FreezeLayers]:
        layer.trainable = False

    #Adding custom Layers
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)


    # x = Dense(1024, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)

    # creating the final model
    model_final = Model(input = model.input, output = predictions)

    # compile the model

    model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

    model_final.summary()

    # Initiate the train ,validion,test generators with data Augumentation
    train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)

    valid_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)#

    test_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    rotation_range=30)

    train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = "categorical")



    validation_generator = valid_datagen.flow_from_directory(
    valid_folder,
    target_size = (img_height, img_width),
    class_mode = "categorical")

    test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size = (img_height, img_width),
    class_mode = "categorical")



    #    Training the model
    model_final.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=1)





    # the final prediction on the test dataset
    y_pred = model_final.predict_generator(test_generator, nb_test_samples, workers=4)


    Images_list = []
    Answers_list = []
    correct = 0
    print(len(test_generator.filenames))
    for i, f in enumerate(test_generator.filenames):
        # print(y_pred[i])
     if y_pred[i][0]>y_pred[i][1]:
            # if the prediction gives ct as an answer
            Answers_list.append('ct')
            Images_list.append(str(f).split('\\')[1][:-4])
            if f.startswith('ct'):
                # if the pridiction is correct
                correct += 1
     else:
            # if the prediction gives mri as an answer


            Answers_list.append('mri')
            Images_list.append((f.split("\\"))[1][:-4])
            if f.startswith('mri'):
                # if the pridiction is correct
                correct += 1


    # save answer as prediction for each Image
    prediction = pd.DataFrame( {'Images': Images_list,'Answers': Answers_list})

    # print the prediction of the test-set
    print('Correct predictions: '+str(correct/len(test_generator.filenames)) , ", num of images: " , len(test_generator.filenames))


    return prediction






# The function returns the predictions for a particular model
def Get_Predition_of_Train_Model():
    batch_size = 10
    epochs =15
    FreezeLayers=25

    # the models
    model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    # model = applications.Xception(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    # model = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    # model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    # model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))


    prediction=Train_Model_And_Predition (model,epochs,batch_size,FreezeLayers)
    return prediction



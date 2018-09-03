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
# train_data_dir = "data/train"
# validation_data_dir = "data/valid"

# dataset_folder_path = 'MRI_CT_data'


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


def Train_Model_And_Predition(model,epochs,batch_size):
    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers[:-25]:
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

    # Initiate the train and test generators with data Augumentation
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



    # Save the model according to the conditions
    checkpoint = ModelCheckpoint("vgg19_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


    model_final.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=1)





    # the final prediction on the test dataset
    y_pred = model_final.predict_generator(test_generator, nb_test_samples, workers=4)

    prediction=pd.DataFrame(columns=['Images', 'Answers'])
    Images_list = []
    Answers_list = []
    correct = 0
    print(len(test_generator.filenames))
    for i, f in enumerate(test_generator.filenames):

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



    prediction = pd.DataFrame( {'Images': Images_list,'Answers': Answers_list})
    # prediction = prediction.set_index('Images')
    # print(prediction.set_index('Images'))

    print('Correct predictions: '+str(correct/len(test_generator.filenames)) , ", num of images: " , len(test_generator.filenames))


    return prediction



def WritingAnswers(prediction):
    dfANS = pd.read_excel(open('FinelFiles/myAnswers.xlsx', 'rb'), names=['Images', 'Questions', 'Answers'])
    dfANS = dfANS.drop(columns=['Answers'])
    dfANS = pd.merge(dfANS, prediction, on=['Images'])
    writer = ExcelWriter('FinelFiles/VQA_TestSet_Res.xlsx')
    dfANS.to_excel(writer, 'ImagesOfCtMriVal', index=False)
    writer.save()





def Writing_Answers_according_the_predictions_of_trained_Model():
    batch_size = 10
    epochs =15

    # model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    # model = applications.Xception(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    # model = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    # model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    my_prediction=Train_Model_And_Predition (model,epochs,batch_size)
    WritingAnswers(my_prediction)

Writing_Answers_according_the_predictions_of_trained_Model()
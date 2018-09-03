from __future__ import print_function

import glob
from keras.preprocessing import image

import Image
import keras
from keras.datasets import cifar10
from keras import backend as K, optimizers
import matplotlib
from matplotlib import pyplot as plt, pyplot
import numpy as np
import ImageToHistogram as img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import applications
from keras import models
from keras import layers

# Input image dimensions
img_rows, img_cols = 224, 224
num_classes=2

if K.image_data_format() == "channels_first":
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)
epochs=10
channels = 3
if (channels == 1):
    color_mode_ = "grayscale"
else:
    color_mode_ = "rgb"


batch_size = 64
num_classes = 2
epochs = 1

dataset_folder_path = 'MRI_CT_data'
train_folder = dataset_folder_path + '/train'
test_folder = dataset_folder_path + '/test'

test_files = glob.glob(test_folder + '/**/*.jpg')
train_files = glob.glob(train_folder + '/**/*.jpg')


x_train = [image.load_img( img, target_size=(224, 224)) for img in train_files]
x_test = [image.load_img( img, target_size=(224, 224)) for img in test_files]

x_train = [np.array(img) for img in x_train]
x_test = [np.array(img) for img in x_test]
# print(len(x_train[0]))
# print(len(x_train[1]))
#
train_examples = len(train_files)
test_examples = len(test_files)
print("Number of train examples: " , train_examples)
print("Number of test examples: ", test_examples)
#
#
# (train), (test) =train_examples,test_examples


# (x_trainB, y_trainB), (x_testB, y_testB) = cifar10.load_data()
# print(len(x_trainB[0]))
# print(len(x_trainB[1]))


x_train=np.asarray(x_train, dtype=None, order=None)
x_test=np.asarray(x_test, dtype=None, order=None)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train /= 255
x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')


# Initialize Generator
datagen = img.ImageDataGenerator(contrast_stretching=True, adaptive_equalization=False, histogram_equalization=False)


train_generator = datagen.flow_from_directory(
    train_folder,
    target_size=(img_rows, img_cols ),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    test_folder,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)



print("len of x train before eugmantation:")
print(len(x_train))

# fit parameters from data
datagen.fit(x_train,True)

print("len of x train after eugmantation:")
print(len(x_train))


train_generator = datagen.flow_from_directory(
    train_folder,
    target_size=(img_rows, img_cols ),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    save_to_dir='MRI_CT_data/aug',
    save_prefix='aug',
    save_format='jpg'
)


validation_generator = datagen.flow_from_directory(
    test_folder,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)



model = Sequential()
model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()


#Load the VGG model

vgg_conv = applications.VGG16(include_top=False, input_shape=input_shape)


# vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=Input(shape=input_shape))


# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
#

#
# # Create the model
model = models.Sequential()
# model.add(Reshape(target_shape=(128, 128, 2), input_shape=list(vgg_conv.output.get_shape().as_list()[1:])))

# load_model
# model = load_model('modelAfterFirstFit.h5')

# # Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

# # # Show a summary of the model. Check the number of trainable parameters
model.summary()



# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])



# datagen.fit(x_train)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1)


#
# packages 2 install
#

# !pip install tqdm
# !conda install -y Pillow


# ---------------------------------------------------------------------
# Load util
import keras
import matplotlib.pyplot as plt
from keras.applications import Xception # TensorFlow ONLY

from keras.applications import VGG16

from keras.applications.resnet50 import ResNet50
import numpy as np
import glob

from keras.layers import Reshape, Convolution2D
from keras.models import Sequential, Model
from keras import optimizers, Input
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import Xception # TensorFlow ONLY

from keras.applications import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50

epochs=30

from keras.models import load_model

img_height = img_width = 224
channels = 3
if (channels == 1):
    color_mode_ = "grayscale"
else:
    color_mode_ = "rgb"

img_rows, img_cols = 224, 224
if K.image_data_format() == "channels_first":
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)

    input_crop = Input(shape=input_shape)
    print(type(input_crop))

# dataset_folder_path = 'MRI_CT_data'
dataset_folder_path = 'data'
train_folder = dataset_folder_path + '/train'
valid_folder = dataset_folder_path + '/valid'
test_folder = dataset_folder_path + '/test'

train_files = glob.glob(train_folder + '/**/*.jpg')
valid_files = glob.glob(valid_folder + '/**/*.jpg')
test_files = glob.glob(test_folder + '/**/*.jpg')

train_examples = len(train_files)
valid_examples = len(valid_files)
test_examples = len(test_files)

print("Number of train examples: " , train_examples)
print("Number of valid examples: ", valid_examples)
print("Number of test examples: ", test_examples)

#   Download and extract the doge and cate pictures.
# ---------------------------------------------------------------------



#Load the VGG model

# vgg_conv = applications.ResNet50(include_top=False, input_shape=input_shape)
# vgg_conv = applications.VGG16(include_top=False, input_shape=input_shape)
# vgg_conv = applications.Xception(include_top=False, input_shape=input_shape)
vgg_conv = applications.InceptionV3(include_top=False, input_shape=input_shape)



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
class_weight = {0: 1.,
                1: 50.,
                2: 2.}




train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        zoom_range=0.2,
        horizontal_flip=True)

# Change the batchsize according to your system RAM
train_batchsize = 100
val_batchsize = 10
test_batchsize = 10


train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(img_height, img_width),
    batch_size=train_batchsize,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    valid_folder,
    target_size=(img_height, img_width),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)
test_generator = test_datagen.flow_from_directory(
   test_folder,
    target_size=(img_height, img_width),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)



# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1)

y_pred = model.predict_generator(test_generator, test_examples//val_batchsize, workers=4)

# print(len(y_pred))
# print(y_pred)

# model.predict_classes(test_x)
# np.count_nonzero(y_pred == test_y)/len(test_y)
MriOrCt=[]
correct = 0
for i, f in enumerate(test_generator.filenames):

    if y_pred[i-2][0]>y_pred[i-2][1]:
        MriOrCt.append('ct')
        print("ct")
        if f.startswith('ct'):
            correct += 1
    if y_pred[i-2][1]>=y_pred[i-2][0]:
        print("mri")
        MriOrCt.append('mri')
        if f.startswith('mri'):
            correct += 1
    # if f.startswith('ct') and y_pred[i-2][0]>y_pred[i-2][1]:
    #     correct +=1
    # if f.startswith('mri') and y_pred[i-2][1]>=y_pred[i-2][0]:
    #     correct +=1

print('Correct predictions: '+str(correct/len(test_generator.filenames)) , ", num of images: " , len(test_generator.filenames))

# adding the correctanswers to the validation file
MriOrCt = np.array(MriOrCt)
MriOrCt = pd.Series(MriOrCt)

writer = pd.ExcelWriter('outputFiles/MRI CT Answers Validation.xlsx', engine='openpyxl')
wb = writer.book
dfANS = dfANS.set_index('Answers')

dfANS['Answers'] = MriOrCt

dfANS.to_excel(writer, index=False)
wb.save('outputFiles/MRI CT Answers Validation for check.xlsx')
# *********************************************

# *************** Up-sample Minority Class ************

from sklearn.utils import resample
# First, we'll separate observations from each class into different DataFrames.
# Next, we'll resample the minority class with replacement, setting the number of samples to match that of the majority class.
# Finally, we'll combine the up-sampled minority class DataFrame with the original majority class DataFrame.

# Separate majority and minority classes
df_majority = dfANS[dfANS.balance == 0]
df_minority = dfANS[dfANS.balance == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=7,  # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled.balance.value_counts()

# ******************* Penalize Algorithms (Cost-Sensitive Training) ************************
from sklearn.metrics import roc_auc_score

# Separate input features (X) and target variable (y)
y = dfANS.balance
X = dfANS.drop('balance', axis=1)

# Train model
clf_3 = SVC(kernel='linear',
            class_weight='balanced',  # penalize
            probability=True)

clf_3.fit(X, y)

# Predict on training set
pred_y_3 = clf_3.predict(X)

# Is our model still predicting just one class?
print(np.unique(pred_y_3))
# [0 1]

# How's our accuracy?
print(accuracy_score(y, pred_y_3))
# 0.688

# What about AUROC?
prob_y_3 = clf_3.predict_proba(X)
prob_y_3 = [p[1] for p in prob_y_3]
print(roc_auc_score(y, prob_y_3))
# 0.5305236678

# *********************** Use Tree-Based Algorithms **************************

from sklearn.ensemble import RandomForestClassifier

# Separate input features (X) and target variable (y)
y = dfANS.balance
X = dfANS.drop('balance', axis=1)

# Train model
clf_4 = RandomForestClassifier()
clf_4.fit(X, y)

# Predict on training set
pred_y_4 = clf_4.predict(X)

# Is our model still predicting just one class?
print(np.unique(pred_y_4))
# [0 1]

# How's our accuracy?
print(accuracy_score(y, pred_y_4))
# 0.9744

# What about AUROC?
prob_y_4 = clf_4.predict_proba(X)
prob_y_4 = [p[1] for p in prob_y_4]
print(roc_auc_score(y, prob_y_4))
# 0.999078798186
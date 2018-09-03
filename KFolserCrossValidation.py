# Load libraries
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from keras import applications
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras.applications import Xception
from sklearn.model_selection import cross_val_score

from keras.applications import VGG16
from keras.applications.inception_v3 import InceptionV3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
import csv
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import glob
img_width, img_height = 256, 256
# Set random seed
np.random.seed(0)

# Number of features
number_of_features = 100

# Generate features matrix and target vector
features, target = make_classification(n_samples = 10000,
                                       n_features = number_of_features,
                                       n_informative = 3,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       weights = [.5, .5],
                                       random_state = 0)

# Create function returning a compiled network
def create_network():
    channels = 3
    batch_size = 10
    epochs = 15
    model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, channels))
    # model = applications.Xception(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    # model = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    # model = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))
    # model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))

    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers[:-10]:
        layer.trainable = False

    # Adding custom Layers
    x = model.output
    x = Flatten()
    x = Dense(50, activation="relu")(x)
    # x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    # x = Dense(1024, activation="relu")(x)
    predictions = Dense(2, activation="softmax")

    # creating the final model
    model_final = Model(input=model.input, output=predictions)

    # compile the model
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                        metrics=["accuracy"])
    return model_final

def loadData():
    df = pd.read_csv('FinelFiles/VQA_TrainingLabelSet.csv')  # open csv file and rename columns

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
    sizeTrain=len(train_files)
    # X_train, X_test, y_train, y_test = train_test_split(train_filesv,train_files,df['balance'], test_size = 0.0, random_state = 0)
    DataImages = [image.load_img("images\Train-images\\" + img + ".jpg", target_size=(256, 256)) for img in df['Images']]
    DataImages =[image.img_to_array(img) for img in DataImages ]
    # DataImages =[img.flatten() for img in DataImages ]
    print("***************************")
    print(type(DataImages))
    print("***************************")

    Label = [row['balance'] for index, row in df.iterrows()]
    Label=np.array(Label)
    DataImages=np.array(DataImages)
    return (DataImages, Label)

# Wrap Keras model so it can be used by scikit-learn
neural_network = KerasClassifier(build_fn=create_network,
                                 epochs=10,
                                 batch_size=100,
                                 verbose=0)


x_train,y_train=loadData()
clf = svm.SVC(kernel='linear', C=1)
# Evaluate neural network using three-fold cross-validation
cross_val_score1=cross_val_score(neural_network,x_train,y_train, cv=10,scoring='accuracy')
print(cross_val_score.mean())
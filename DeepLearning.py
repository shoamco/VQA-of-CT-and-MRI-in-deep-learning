from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow
import numpy as np
import pydot
from matplotlib import pyplot
import pandas as pd
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




from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np


# def CALC_SHOW_PCA_3D(Data, STRING_MODEL):
#     pca = PCA(n_components=3)
#     DataPCA = pca.fit(Data).transform(Data)
#     target_names = ['MRI', 'CT']
#     group1 = np.full((len(ImagesOfMri)), 0)#Mri Group=0
#     group2 = np.full((len(ImagesOfCt)), 1)#ct Group=0
#     Group = np.concatenate((group1, group2), axis=0, out=None)#all Groups
#     x = [x for (x ,y , z) in DataPCA]
#     y = [y for (x ,y , z) in DataPCA]
#     z=[z for (x ,y , z) in DataPCA]
#
#     fig = pyplot.figure()
#     ax = Axes3D(fig)
#     colors = ['red', 'blue']
#     lw = 2
#
#     for color, i, target_name in zip(colors, [0, 1], target_names):#Draw a 3D graph by the colored groups
#       ax.scatter(DataPCA[Group == i, 0], DataPCA[Group == i, 1], DataPCA[Group == i, 2], color=color, alpha=.8, lw=lw, label=target_name)
#     plt.legend(loc='best', shadow=False, scatterpoints=1)
#     plt.title('PCA of mri/ct '+STRING_MODEL)
#
#     pyplot.show()

# model = VGG16(weights='imagenet', include_top=False)
# model = ResNet50(weights='imagenet')
# model = VGG19(weights='imagenet')
model = densenet.DenseNet169(weights='imagenet')


def get_features_images(images):
    features=[]
    for img in images:
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model.predict(x)[0])
    return features




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

ImagesMri = [image.load_img("images\Train-images\\" + img + ".jpg", target_size=(224, 224)) for img in ImagesOfMri]
ImagesCt = [image.load_img("images\Train-images\\" + img + ".jpg", target_size=(224, 224)) for img in ImagesOfCt]




featuresMri=get_features_images(ImagesMri)
featuresCt=get_features_images(ImagesCt)

sizeMri=len(featuresMri)
sizeCt=len(featuresCt)


TrainingData = np.concatenate((featuresMri[:int(sizeMri*0.8)], featuresCt[:int(sizeCt*0.8)]), axis=0, out=None)
y = np.full((len(featuresMri[:int(sizeMri*0.8)])), 0)
y1 = np.full((len(featuresCt[:int(sizeCt*0.8)])), 1)
Y = np.concatenate((y, y1), axis=0, out=None)
# print(len(featuresCt))






sizefeatur=len(featuresCt[0])

# create model
model = Sequential()
model.add(Dense(12, input_dim=sizefeatur, init='uniform', activation='relu'))
model.add(Dense(sizefeatur, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model-The training proces
model.fit(TrainingData, Y, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(TrainingData)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


y = np.full((len(featuresMri[int(sizeMri*0.8):])), 0)
y1 = np.full((len(featuresCt[int(sizeCt*0.8):])), 1)
Yreal= np.concatenate((y, y1), axis=0, out=None)

# new instances where we do not know the answer
newData= np.concatenate((featuresMri[int(sizeMri*0.8):], featuresCt[int(sizeCt*0.8):]), axis=0, out=None)

# make a prediction-predict the class
ynew = model.predict_classes(newData)
# show the inputs and predicted outputs
#
for i in range(len(newData)):
	# print("X=%s, Predicted=%s" % (newData[i], ynew[i]))
	print("Real=%s,Predicted=%s" % (Yreal[i], ynew[i]))
    # if Yreal[i]==ynew[i]:
    #     n+=1
n=[1 for i in range(len(newData)) if Yreal[i]==ynew[i]]
correct=sum(n)
print("%s /%s Are correct "%(correct,len(newData)))








# CALC_SHOW_PCA_3D(dataFeatures,"DL-VGG16")



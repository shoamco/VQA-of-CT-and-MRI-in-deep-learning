
import numpy as np
from keras import applications
import os
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

# dimensions of our images.
img_width, img_height,channels = 224, 224,3

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/valid'
lenImgTrain=len(os.listdir('data/train/ct')+os.listdir('data/train/mri'))
lenImgValid=len(os.listdir('data/valid/ct')+os.listdir('data/valid/mri'))

print("lenImgTrain "+str(lenImgTrain))
print("lenImgValid "+str(lenImgValid))

nb_train_samples =lenImgTrain
nb_validation_samples = lenImgValid
epochs = 50
batch_size =2



datagen = ImageDataGenerator(rescale=1. / 255)
# ************************************************************************
# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, channels))

validationGenerator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True)
trainGenerator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True)

bottleneck_features_train = model.predict_generator(trainGenerator, nb_train_samples // batch_size)
np.save(open('bottleneck_features_train2', 'wb'), bottleneck_features_train)


bottleneck_features_validation = model.predict_generator(validationGenerator, nb_validation_samples // batch_size)

np.save(open('bottleneck_features_validation2', 'wb'), bottleneck_features_validation)

################################################33

train_data = np.load('bottleneck_features_train2','r+',encoding='latin1')

train_labels = np.array(
        [0] *(int (nb_train_samples / 2)) + [1] *(int(nb_train_samples / 2)))

validation_data = np.load('bottleneck_features_validation2','r+',encoding='latin1')

validation_labels = np.array(
        [0] * int((nb_validation_samples / 2)) + [1] * (int(nb_validation_samples / 2)))


##########Transfer Learning ###########

model = applications.VGG16(include_top=False, weights='imagenet')
model.summary()

model = Sequential()
model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(bottleneck_features_train, train_labels, epochs=15, batch_size=batch_size)

# *************** PART 2 **********************************

vgg16 = applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, channels))
combinedModel = Model(inputs= vgg16.input, outputs= model(vgg16.output))

for layer in combinedModel.layers[:-3]:
    layer.trainable = False
combinedModel.summary()
model.save_weights('fc_model.h5')
combinedModel.compile(loss='binary_crossentropy',
              optimizer = optimizers.RMSprop(lr=1e-4, decay=0.9), # optimizers.SGD(lr=1e-4, momentum=0.9)
              metrics=['accuracy'])

# prepare data augmentation configuration


combinedModel.fit_generator(
    trainGenerator,
    steps_per_epoch=nb_train_samples //batch_size,
    epochs=2,
    validation_data=validationGenerator,
    validation_steps=nb_validation_samples //batch_size) # len(valid_generator.filenames)



predictClasses = model.predict_classes(validation_data )
# show the inputs and predicted outputs

n=[1 for i in range(len(validation_labels)) if validation_labels[i]==predictClasses[i]]
correct=sum(n)
print("%s /%s Are correct "%(correct,len(validation_data)))
print("%s Are correct "%(correct/len(validation_data)))

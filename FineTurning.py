from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model
from keras.models import InputLayer
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

# path to the model weights files.
weights_path = 'bottleneck_fc_model.h5'
top_model_weights_path='bottleneck_fc_model.h5'
# weights_path = '../keras/examples/vgg16_weights.h5'
# top_model_weights_path = 'fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150


train_data_dir = 'data/train'
validation_data_dir = 'data/valid'
nb_train_samples =16
nb_validation_samples = 4
epochs = 50
batch_size = 16
# build the VGG16 network
# model = applications.VGG16(weights='imagenet', include_top=False)
# model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
#
# print('Model loaded.')
#
# # build a classifier model to put on top of the convolutional model
# model = Sequential()
# print(model.output_shape[1:])
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# # note that it is necessary to start with a fully-trained
# # classifier, including the top classifier,
# # in order to successfully do fine-tuning
# model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
# model.add(top_model)

model=load_model('model.h5')
model.summary()
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
model.summary()
# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')
# model.fit_generator(train_generator)
# print(train_generator[0])
# fine-tune the model
# model.fit(train_generator[0],train_generator.classes)
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples
    ,epochs=epochs
    # # batch_size=batch_size
    # ,validation_data=validation_generator,
    # nb_val_samples=nb_validation_samples
)


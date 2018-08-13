from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import glob
img_width, img_height = 256, 256
# train_data_dir = "data/train"
# validation_data_dir = "data/valid"

# dataset_folder_path = 'MRI_CT_data'
dataset_folder_path = 'data'
train_data_dir = dataset_folder_path + '/train'
validation_data_dir = dataset_folder_path + '/test'


train_files = glob.glob(train_data_dir + '/**/*.jpg')
test_files = glob.glob(validation_data_dir + '/**/*.jpg')


nb_train_samples = len(train_files)
nb_validation_samples = len(test_files)
print("Number of train examples: " , nb_train_samples)
print("Number of test examples: ", nb_validation_samples)
channels = 3
batch_size = 10
epochs = 30

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, channels))


# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:5]:
    layer.trainable = False

#Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
# x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model
model_final = Model(input = model.input, output = predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

# Save the model according to the conditions
checkpoint = ModelCheckpoint("vgg19_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Train the model
# model_final.fit_generator(
# train_generator,
# samples_per_epoch = nb_train_samples,
# epochs = epochs,
# validation_data = validation_generator,
# nb_val_samples = nb_validation_samples,
# callbacks = [checkpoint, early])


model_final.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1)
model_final.summary()

y_pred = model.predict_generator(validation_generator, nb_validation_samples//batch_size, workers=4)
print(len(y_pred))
print(y_pred)
# print(len(y_pred[0]))
# print(type(y_pred[0]))
# model.predict_classes(test_x)
# np.count_nonzero(y_pred == test_y)/len(test_y)

# correct = 0
# for i, f in enumerate(validation_generator.filenames):
#     print(i)
#     # TODO if [0]>[1]
#     if f.startswith('ct') and y_pred[i-2][0]>y_pred[i-2][1]:
#         correct +=1
#     if f.startswith('mri') and y_pred[i-2][1]>=y_pred[i-2][0]:
#         correct +=1
#
# print('Correct predictions: '+str(correct/len(validation_generator.filenames)) , ", num of images: " , len(validation_generator.filenames))


#
# packages 2 install
#

# !pip install tqdm
# !conda install -y Pillow


# ---------------------------------------------------------------------
# Load util
import matplotlib.pyplot as plt

import numpy as np
import glob

from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator

dataset_folder_path = 'MRI_CT_data'
train_folder = dataset_folder_path + '/train'
test_folder = dataset_folder_path + '/test'

test_files = glob.glob(test_folder + '/**/*.jpg')
train_files = glob.glob(train_folder + '/**/*.jpg')

train_examples = len(train_files)
test_examples = len(test_files)
print("Number of train examples: " , train_examples)
print("Number of test examples: ", test_examples)

#   Download and extract the doge and cate pictures.
# ---------------------------------------------------------------------


from keras.preprocessing.image import ImageDataGenerator

"""View some sample images:"""

datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        zoom_range=0.2,
        horizontal_flip=True)


# ---------------------------------------------------------------------
# 2. Display 5 random images
# ---------------------------------------------------------------------
img_height = img_width = 200
channels = 1
if (channels == 1):
    color_mode_ = "grayscale"
else:
    color_mode_ = "rgb"

#
# train_generator = datagen.flow_from_directory(
#     train_folder,
#     color_mode = color_mode_,
#     target_size=(img_height, img_width),
#     batch_size=1,
#     class_mode=None)


"""## Convolution Neural Networks (CNN)"""

model = Sequential()


# TODO: Add a CNN:
# Note 1: The input_shape needs to be specified in this case (input_height, input_width, channels)
# Note 2: The order usually goes Conv2D, Activation, MaxPool,
# Note 3: Must be flattened before passing onto Dense layers
# Note 4: The loss is binary_crossentropy

model.add(Conv2D(8, kernel_size=(3,3), padding='same', input_shape = (img_width,img_height,channels)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(3, 3)))

model.add(Conv2D(16, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# optimizer='rmsprop'
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

#------------------------------------------------------------------------------
# Training  
#------------------------------------------------------------------------------

batch_size = 20
epoch_num = 1
train_generator = datagen.flow_from_directory(
    train_folder,
    color_mode = color_mode_,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle = True,
    class_mode='binary')

model.fit_generator(train_generator, train_examples//batch_size, epochs=epoch_num)

batch_size = 1
test_generator = datagen.flow_from_directory(
    test_folder,
    color_mode = color_mode_,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary', 
    shuffle=False)
y_pred = model.predict_generator(test_generator, test_examples//batch_size, workers=4)
# model.predict_classes(test_x)
# np.count_nonzero(y_pred == test_y)/len(test_y)

correct = 0
for i, f in enumerate(test_generator.filenames):
    if f.startswith('ct') and y_pred[i]<0.5:
        correct +=1
    if f.startswith('mri') and y_pred[i]>=0.5:
        correct +=1

print('Correct predictions: '+str(correct/len(test_generator.filenames)) , ", num of images: " , len(test_generator.filenames))

#------------------------------------------------------------------------------
# plot some images
#------------------------------------------------------------------------------
batch_size = 5
test_generator = datagen.flow_from_directory(
    test_folder,
    color_mode = color_mode_,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True)

x_test, y_test = next(test_generator)

p = model.predict(x_test)
p = np.hstack([y_pred, 1-y_pred])
label_dict = {0: 'ct', 1: 'mri'}

plt.figure(figsize=(12,12))

for i in range(batch_size):
    print(i)

    plt.subplot(batch_size,2,2*i+1)
    # plt.imshow(x_test[i] , cmap='gray')
    plt.imshow(np.squeeze(x_test[i], axis=2) , cmap='gray')

    plt.title(label_dict[y_test[i]])
    
    plt.subplot(batch_size,2,2*i+2)
    plt.bar(range(2),p[i])
    plt.xticks(range(2), [label_dict[0], label_dict[1]])

plt.show()


from sklearn.metrics import confusion_matrix
loss, acc = model.evaluate(x=x_test, y=y_test)
print(loss, acc)
targets = np.argmax(y_test, axis=-1)
probabilities = model.predict(x=x_test)
predictions = np.argmax(probabilities, axis=-1)
print("targets: " ,targets)
print("predictions: " ,predictions)

# cm = confusion_matrix(y_true=targets, y_pred=predictions)
# print(cm)



#save model
model.save("modelAfterFirstFit.h5")
#

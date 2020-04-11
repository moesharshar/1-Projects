# Part_1 - Building the CNN
# Importing the keras libraries and packages
from docutils.nodes import classifier
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Initiating the CNN
classifier = Sequential()

# step #1 - Add Convolution Layer
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# step #2 - Add Pooling Layer
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Adding a second Convolutional Layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))


# step #3 - Flattening Pooled Feature maps
classifier.add(Flatten())

# step #4 - Building up Fully connected Network
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part_2 Fitting the CNN to the Images
from keras.preprocessing.image import ImageDataGenerator
from keras import models
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32,
                                                 class_mode='binary')

testing_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32,
                                               class_mode='binary')

classifier.fit_generator(training_set, samples_per_epoch=8000, nb_epoch=25, validation_data=testing_set,
                         nb_val_samples=2000)

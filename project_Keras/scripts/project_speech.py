# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:52:39 2021

@author: Stasiu
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import SGD
from keras.layers import MaxPooling2D
from keras import utils

import librosa
import librosa.display
import random


num_of_samples = 3000
num_of_test_samples = 300
batch_size = 64
no_epochs = 50
input_shape = 64

data_dir = 'data/'

dataset = []

### PREPARING THE DATASET ###

for i in range(10):
    directory = data_dir + str(i) + "/" + str(i)
    
    for j in range(1, num_of_samples + 1):
      y, sr = librosa.load(directory + " (" + str(j) + ").wav")
      no_samples = len(y)
      #spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length= math.floor(no_samples/128.))
      spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=input_shape, hop_length= math.floor(no_samples/input_shape))
      dataset.append( (spectrogram, i) )
      

random.shuffle(dataset)

data_train = dataset[:(num_of_samples - num_of_test_samples)*10]
data_test = dataset[(num_of_samples - num_of_test_samples)*10:]


X_train, y_train = zip(*data_train)
X_test, y_test = zip(*data_test)


# Reshape the input
X_train = np.array([np.resize(x, (input_shape, input_shape)).reshape((input_shape, input_shape, 1)) for x in X_train])
X_test = np.array([np.resize(x, (input_shape, input_shape)).reshape((input_shape, input_shape, 1)) for x in X_test])


# One-Hot encoding for classes <- CHECK
y_train = np.array(utils.to_categorical(y_train, 10))
y_test = np.array(utils.to_categorical(y_test, 10))



# INTRODUCING MODEL #

voice_model = Sequential()

voice_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(input_shape, input_shape, 1)))
voice_model.add(MaxPooling2D(pool_size=(2, 2)))
voice_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
voice_model.add(MaxPooling2D(pool_size=(2, 2)))
voice_model.add(Flatten())
voice_model.add(Dense(512, activation='relu'))
voice_model.add(Dense(10, activation='softmax'))


voice_model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

### FITING ###
voice_model_info = voice_model.fit(
	x=X_train, 
	y=y_train,
    epochs=no_epochs,
    batch_size=batch_size,
    validation_data= (X_test, y_test))


voice_model.summary()

"""
voice_model_info = voice_model.fit_generator(
    x=X_train, 
	y=y_train,
    steps_per_epoch = num_of_samples // batch_size,
    epochs = no_epochs,
    validation_data = (X_test, y_test),
    validation_steps = num_of_test_samples // batch_size)

#voice_model.save('model_MNIST.h5')
"""

# VISUALISATION
fig_1 = plt.figure(1, dpi = 300)
ax1 = fig_1.add_subplot()
ax1.grid(True)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.set_title("MNIST voice model, MEL64")
ax1.plot(voice_model_info.history["accuracy"] , "bs", label = "Training")
ax1.plot(voice_model_info.history["val_accuracy"] , "r^", label = "Validation")

ax1.legend()



fig_2 = plt.figure(2, dpi = 300)
ax2 = fig_2.add_subplot()
ax2.grid(True)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.set_title("MNIST voice model, MEL64")
ax2.plot(voice_model_info.history["loss"] , "r^", label = "Validation")


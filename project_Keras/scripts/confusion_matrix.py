# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:24:47 2020

@author: Stasiu
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from keras.optimizers import Adam
from keras.models import load_model
from keras import utils
import librosa
import librosa.display
import math


num_of_samples = 3000
batch_size = 64
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

X_test, y_test = zip(*dataset)
X_test = np.array([np.resize(x, (input_shape, input_shape)).reshape((input_shape, input_shape, 1)) for x in X_test])
y_test = np.array(utils.to_categorical(y_test, 10))

# load model
voice_model = load_model('saved_models/model_MNIST_smallD.h5')

# summarize model.
voice_model.summary()

# evaluate the model
voice_model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
score = voice_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (voice_model.metrics_names[1], score[1]*100))



#prediction = voice_model.predict(input_data)



#Confution Matrix and Classification Report

y_pred = voice_model.predict(X_test)   # predict results
print('Confusion Matrix')
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print('Classification Report')
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

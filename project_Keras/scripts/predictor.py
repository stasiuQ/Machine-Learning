# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:24:47 2020

@author: Stasiu
"""

import numpy as np

import tensorflow as tf
from keras.optimizers import Adam
from keras.models import load_model
import librosa
import librosa.display
import math


# load sample data
input_shape = 64

data_dir = 'data/mine/'


# load model
voice_model = load_model('saved_models/model_MNIST_MEL64.h5')

# summarize model.
#voice_model.summary()

# evaluate the model
voice_model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])



i = 2   # considered number

### PREPARING THE DATASET ###


y, sr = librosa.load(data_dir + str(i) + ".wav")
no_samples = len(y)
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=input_shape, hop_length= math.floor(no_samples/input_shape))
img = librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time')

input_data = np.array([np.resize(spectrogram, (input_shape, input_shape)).reshape( (input_shape, input_shape, 1) )])

prediction = voice_model.predict(input_data)
print(prediction)



#score = voice_model.evaluate(validation_generator, verbose=0)
#print("%s: %.2f%%" % (emotion_model.metrics_names[1], score[1]*100))


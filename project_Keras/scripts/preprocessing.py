# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:05:05 2021

@author: Stasiu
"""
import matplotlib.pyplot as plt
import numpy as np
import librosa
import random
import math





num_of_samples = 2
num_of_test_samples = 1
batch_size = 32
no_epochs = 50
input_shape = (128, 128, 1)

data_dir = 'data/'

dataset = []

### PREPARING THE DATASET ###

for i in range(10):
    directory = data_dir + str(i) + "/" + str(i)
    
    for j in range(1, num_of_samples + 1):
      y, sr = librosa.load(directory + " (" + str(j) + ").wav")
      no_samples = len(y)
      spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length= math.floor(no_samples/128.))
      img = librosa.display.specshow(spectrogram)
      dataset.append( (spectrogram, i) )
      

random.shuffle(dataset)

data_train = dataset[:(num_of_samples - num_of_test_samples)*10]
data_test = dataset[(num_of_samples - num_of_test_samples)*10:]


X_train, y_train = zip(*data_train)
X_test, y_test = zip(*data_test)


# Reshape the input
X_train = np.array([np.resize(x, (128, 128)).reshape(input_shape) for x in X_train])
X_test = np.array([np.resize(x, (128, 128)).reshape(input_shape) for x in X_test])
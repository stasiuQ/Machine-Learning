# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:05:05 2021

@author: Stasiu
"""
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import random
import math



input_shape = (128, 128, 1)

data_dir = 'data/'

i = 6   # number
j = 54  # speaker

### PREPARING THE DATASET ###
fig, ax = plt.subplots(dpi=300, nrows=2, ncols=1, sharex=True)

for i in range(2):
    directory = data_dir + str(i) + "/" + str(i)
    
    y, sr = librosa.load(directory + " (" + str(j) + ").wav")
    no_samples = len(y)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length= math.floor(no_samples/128.))
    
    img = librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time', ax=ax[i])
    ax[i].set(title= (str(i) + " digit"))
    ax[i].label_outer()



  

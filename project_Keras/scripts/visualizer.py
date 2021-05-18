# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:24:47 2020

@author: Stasiu
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.optimizers import Adam
from keras.models import load_model
from keras import models
import librosa
import librosa.display
import math


# load sample data
input_shape = 64
no_channel = 18

data_dir = 'data/mine/'
i = 0    # chosen number


### PREPARING DATA ###
y, sr = librosa.load(data_dir + str(i) + ".wav")
no_samples = len(y)

plt.figure(dpi=150)

spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length= math.floor(no_samples/input_shape))
img = librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time')

input_data = np.array([np.resize(spectrogram, (input_shape, input_shape)).reshape((input_shape, input_shape, 1))])



# load model
voice_model = load_model('saved_models/model_MNIST_smallD.h5')
# summarize model.
voice_model.summary()
# evaluate the model
voice_model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])


layer_outputs = [layer.output for layer in voice_model.layers[:4]]  # Extracts the outputs of CNN layers
activation_model = models.Model(inputs=voice_model.input, outputs=layer_outputs) 
# Creates a model that will return these outputs, given the model input

activations = activation_model.predict(input_data) # Returns a list of five Numpy arrays: one array per layer activation


"""
first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, no_channel], cmap='viridis')
#score = voice_model.evaluate(validation_generator, verbose=0)
#print("%s: %.2f%%" % (emotion_model.metrics_names[1], score[1]*100))
"""


layer_names = []
for layer in voice_model.layers[:4]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]), dpi=300)
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

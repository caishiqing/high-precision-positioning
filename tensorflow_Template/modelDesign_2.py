import scipy.io as sio                        
import h5py 
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def Model_2(input_shape, output_shape):
    
    inputs = keras.Input(shape = input_shape)
    x = layers.Conv2D(256, kernel_size = 2, strides = 1, activation = 'relu', padding= 'same')(inputs)
    x = layers.MaxPool2D(pool_size = (2, 1), strides= (2, 1), padding = 'valid')(x)
    x = layers.Conv2D(512, kernel_size = 2, strides = 1, activation = 'relu', padding = 'same')(x)
    x = layers.MaxPool2D(pool_size = (2, 1), strides= (2, 1), padding = 'valid')(x)
    x = layers.Conv2D(768, kernel_size = 2, strides = 1, activation = 'relu', padding= 'same')(x)
    x = layers.MaxPool2D(pool_size = (2, 1), strides= (2, 1), padding = 'valid')(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_shape)(x)
    
    model = keras.Model(inputs = inputs, outputs = outputs, name = 'CNN')
    return model
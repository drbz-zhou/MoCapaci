# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:48:15 2021

@author: bzhou
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_TConv(filters = 40, kernel = (10,4), dense = 100, numClass = 20):
    model = keras.models.Sequential([
        layers.Conv2D( filters = filters, kernel_size = kernel, padding='same', activation='relu', input_shape=(400,4,1)),
        layers.MaxPooling2D(pool_size=(10, 1)),
        layers.Dropout(0.2),
        layers.Conv2D( filters = filters, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(10, 1)),
        layers.Dropout(0.2),
        layers.Conv2D( filters = filters, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        #layers.MaxPooling2D(pool_size=(10, 1)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(dense, activation='relu'),
        layers.Dense(numClass, activation='softmax')
    ])
    return model
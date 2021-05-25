# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:48:15 2021

@author: bzhou
"""
from tensorflow import keras
from tensorflow.keras import layers
import ModelBuilder_TransformerEncoder as MB_TF

def retur_model(model_type = 'TConv'):
    if model_type == 'TConv':
        model = build_TConv()
    elif model_type == 'LSTM':
        model = build_LSTM()
    #not in use because need to pass the parameters
    return model

def build_TConv(filters = 40, kernel = (10,4), dense = 100, numClass = 20):
    model = keras.models.Sequential([
        #layers.AveragePooling2D(pool_size=(5, 1), strides=(2,1), padding='same', input_shape=(400,4,1)),
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
        layers.Dropout(0.2),
        layers.Dense(numClass, activation='softmax')
    ])
    return model

def build_LSTM(lstm_units = 100, dense = 100, numClass = 20):
    model = keras.models.Sequential([
        # Shape [batch, time, features] => [batch, 50, 5903]
        layers.AveragePooling2D(pool_size=(5, 1), strides=(2,1), padding='same', input_shape=(400,4,1)),
        layers.Reshape((200, 4), input_shape=(200,4,1)),
        layers.Bidirectional(layers.LSTM(lstm_units), input_shape=(200,4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(dense, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(numClass, activation='softmax')
    ])
    return model

def build_Conv_LSTM(conv_filters = 40, conv_kernel = (10,4), lstm_units = 100, dense = 100, numClass = 20):
    model = keras.models.Sequential([
        # Shape [batch, time, features] => [batch, 50, 5903]
        layers.Conv2D( filters = conv_filters, kernel_size = conv_kernel, padding='same', activation='relu', input_shape=(400,4,1)),
        layers.Reshape((400, 4*conv_filters), input_shape=(400,4,conv_filters)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(lstm_units), input_shape=(400,4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(dense, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(numClass, activation='softmax')
    ])
    return model

def build_TfEncoder(batch):
    return MB_TF.get_model(batch)
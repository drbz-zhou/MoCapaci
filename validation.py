# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:59:36 2021

@author: bzhou
"""

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import scipy.io
import pickle
import toolbox as tools

from sklearn.metrics import confusion_matrix

modelsavefile = 'model/model1.h5'
outfolder = 'outputs/'

m_person = 3
numClass = 20

X=np.load('data/P'+str(m_person)+'_X.npy')
y=np.load('data/P'+str(m_person)+'_y.npy')
y=y-1
#expanding dim only needed if channels are taken into the 2D format, virtually adding 1 channel at the end of dims
X=np.expand_dims(X,3)
y_cat=tools.label2categorical(y,numClass) #from (windows,1) to (windows,20)
#train-valid-test
ind_train = range(0,300)
ind_valid = range(300,320)
ind_test  = range(320,400)
X_train = X[ind_train, :, :, :]
X_valid = X[ind_valid, :, :, :]
X_test  = X[ind_test,  :, :, :]

y_train = y_cat[ind_train,:]
y_valid = y_cat[ind_valid,:]
y_test  = y[ind_test]

#%% P1 train, P2 test, seen one session by early stopping

m_person = 1
X=np.load('data/P'+str(m_person)+'_X.npy')
y=np.load('data/P'+str(m_person)+'_y.npy')
y=y-1
X=np.expand_dims(X,3)
y_cat=tools.label2categorical(y,numClass) #from (windows,1) to (windows,20)

X_train = X
y_train = y_cat

m_person = 2
X=np.load('data/P'+str(m_person)+'_X.npy')
y=np.load('data/P'+str(m_person)+'_y.npy')
y=y-1
X=np.expand_dims(X,3)
y_cat=tools.label2categorical(y,numClass) #from (windows,1) to (windows,20)

ind_valid = range(0,80)
ind_test  = range(80,400)
X_valid = X[ind_valid, :, :, :]
X_test  = X[ind_test,  :, :, :]
y_valid = y_cat[ind_valid,:]
y_test  = y[ind_test]
#%% P1 train, P2 test, stranger

m_person = 1
X=np.load('data/P'+str(m_person)+'_X.npy')
y=np.load('data/P'+str(m_person)+'_y.npy')
y=y-1
X=np.expand_dims(X,3)
y_cat=tools.label2categorical(y,numClass) #from (windows,1) to (windows,20)

ind_train = range(0,400)
ind_valid = range(320,400)

X_train = X[ind_train, :, :, :]
X_valid = X[ind_valid, :, :, :]
y_train = y_cat[ind_train,:]
y_valid = y_cat[ind_valid,:]

m_person = 2
X=np.load('data/P'+str(m_person)+'_X.npy')
y=np.load('data/P'+str(m_person)+'_y.npy')
y=y-1
X=np.expand_dims(X,3)
y_cat=tools.label2categorical(y,numClass) #from (windows,1) to (windows,20)
X_test  = X
y_test  = y

#%% prepare model
filters = 40
kernel = (40,4)
dense = 100
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
m_opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=m_opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
#%% train model
acc = []
val_acc = []
loss = []
val_loss = []
patience = 1000
epoch = 10000
batch = 64
model, history = tools.train_step(model, epoch, X_train, y_train, X_valid, y_valid, 
                                  modelsavefile, Patience = patience, batch_size = batch)
acc, val_acc, loss, val_loss = tools.append_history(history, acc, val_acc, loss, val_loss)
tools.plot_acc_loss(acc, val_acc, loss, val_loss, file_path=outfolder)
model.load_weights(modelsavefile) #model needs to be built first 
y_predict = model.predict(X_test)
acc_test = sum(y_test == np.argmax(y_predict, axis=1)) / y_test.shape[0]
cm = confusion_matrix(y_test, np.argmax(y_predict, axis=1))
tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path=outfolder)
print(acc_test)
print(cm)
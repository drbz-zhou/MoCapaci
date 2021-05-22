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
import ModelBuilder as MB
import DataProvider as DP

from sklearn.metrics import confusion_matrix

outfolder = 'outputs/'
model_type = 'TConv'
modelsavefile = 'model/'+model_type+'.h5'


m_person = 3
numClass = 20

X_train, X_valid, X_test, y_train, y_valid, y_test = DP.Group_LeaveRecOut(list(range(1,7)))

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
model = MB.build_TConv(filters = 40, kernel = (40,4), dense=100)
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

#%% test
model.load_weights(modelsavefile) #model needs to be built first 
y_predict = model.predict(X_test)
acc_test = sum(y_test == np.argmax(y_predict, axis=1)) / y_test.shape[0]
cm = confusion_matrix(y_test, np.argmax(y_predict, axis=1))
tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path=outfolder,acc=acc_test)
print(acc_test)
print(cm)
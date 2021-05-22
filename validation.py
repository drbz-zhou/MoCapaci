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
numClass = 20

# prepare data
# leave recording out
X_train, X_valid, X_test, y_train, y_valid, y_test = DP.Group_LeaveRecOut(list(range(1,7)))
# leave persons out
X_train, X_valid, X_test, y_train, y_valid, y_test = DP.LeavePersonOut_Stranger(list(range(1,6)),[6])
# leave persons out 
X_train, X_valid, X_test, y_train, y_valid, y_test = DP.LeavePersonOut_Acquaintance(list(range(1,6)),[6])

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
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:19:50 2021

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
m_population = 6 
cm_all = np.zeros((numClass, numClass, m_population))
for m_test in range(m_population):
    # training index is population removing test subject
    m_train = list(range(m_population))
    m_train.pop(m_test)
    
    # prepare data
    X_train, X_valid, X_test, y_train, y_valid, y_test = DP.LeavePersonOut_Stranger(m_train,[m_test])

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
    patience = 100
    epoch = 10000
    batch = 80
    model, history = tools.train_step(model, epoch, X_train, y_train, X_valid, y_valid, 
                                      modelsavefile, Patience = patience, batch_size = batch)
    acc, val_acc, loss, val_loss = tools.append_history(history, acc, val_acc, loss, val_loss)
    tools.plot_acc_loss(acc, val_acc, loss, val_loss, file_path=outfolder+'LPO_'+str(m_test)+'_')
    #%% test
    model.load_weights(modelsavefile) #model needs to be built first 
    y_predict = model.predict(X_test)
    acc_test = sum(y_test == np.argmax(y_predict, axis=1)) / y_test.shape[0]
    cm = confusion_matrix(y_test, np.argmax(y_predict, axis=1))
    tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path=outfolder+'LPO_'+str(m_test)+'_',acc=acc_test)
    cm_all[:,:,m_test] = cm
    
    tools.save_history(outfolder, acc, val_acc, loss, val_loss, cm)
    
    print(acc_test)
    print(cm)

cm = np.sum(cm_all,2)
acc = np.sum(cm*np.eye(numClass, numClass)) / np.sum(cm)
tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path=outfolder,acc=acc)
print(acc)
print(cm)
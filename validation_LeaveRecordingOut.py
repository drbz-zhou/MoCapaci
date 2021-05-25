# -*- coding: utf-8 -*-
"""
Created on Sat May 22 21:50:49 2021

@author: bzhou
"""
from tensorflow import keras

import numpy as np
import toolbox as tools
import ModelBuilder as MB
import DataProvider as DP

from sklearn.metrics import confusion_matrix

session = tools.tf_mem_patch()

outfolder = 'outputs/'
model_type = 'TfEncoder'  # TConv, LSTM, Conv_LSTM, TfEncoder
modelsavefile = 'model/'+model_type+'.h5'
numClass = 20
m_population = 6 
cm_all = np.zeros((numClass, numClass, 0))
batch = 120

for m_rec in range(m_population):
    # leave recording out
    X_train, X_valid, X_test, y_train, y_valid, y_test = DP.Group_LeaveRecOut(list(range(m_population)), m_rec)
    
    #%% prepare model
    
    if model_type == 'TConv':
        model = MB.build_TConv(filters = 40, kernel = (40,4), dense=100)
    elif model_type == 'LSTM':
        model = MB.build_LSTM(lstm_units = 40, dense=100)
    elif model_type == 'Conv_LSTM':
        model = MB.build_Conv_LSTM(conv_filters = 20, conv_kernel = (40,4), lstm_units = 40, dense = 100, numClass = 20)
    elif model_type == 'TfEncoder':
        model = MB.build_TfEncoder(batch)
    # optimizer
    m_opt = keras.optimizers.Adam(learning_rate=0.0005)
    #m_opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0001)
    #m_opt = keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.0001)
    model.compile(optimizer=m_opt,
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    #%% train model
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    patience = 1000
    epoch = 50000
    model, history = tools.train_step(model, epoch, X_train, y_train, X_valid, y_valid, 
                                      modelsavefile, Patience = patience, batch_size = batch)
    acc, val_acc, loss, val_loss = tools.append_history(history, acc, val_acc, loss, val_loss)
    tools.plot_acc_loss(acc, val_acc, loss, val_loss, file_path=outfolder+'LeaveRecOut_'+str(m_rec)+'_'+model_type)
    #%% test
    model.load_weights(modelsavefile) #model needs to be built first 
    y_predict = model.predict(X_test)
    
    acc_test = sum(y_test == np.argmax(y_predict, axis=1)) / y_test.shape[0]
    cm = confusion_matrix(y_test, np.argmax(y_predict, axis=1))
    tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path=outfolder+'LeaveRecOut_'+str(m_rec)+'_'+model_type,acc=acc_test)
    cm_all = np.concatenate( (cm_all, np.expand_dims(cm,2)), 2)
    tools.save_history(outfolder, acc, val_acc, loss, val_loss, cm)
    
    print(acc_test)
    print(cm)
cm = np.sum(cm_all,2)
acc = np.sum(cm*np.eye(numClass, numClass)) / np.sum(cm)
tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path=outfolder+model_type,acc=acc)
print(acc)
print(cm)
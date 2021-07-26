# -*- coding: utf-8 -*-
"""
Created on Mon May 24 00:30:16 2021

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

session = tools.tf_mem_patch()

outfolder = 'outputs/'
model_type = 'TConv'  # Cov1D, TConv, LSTM, Conv_LSTM, TfEncoder, Conv1D_LSTM, ResConv1D
modelsavefile = 'model/'+model_type+'.h5'
numClass = 21
m_population = 1
numRec = 5
batch = 120
cm_all = np.zeros((numClass, numClass, 0))

for m_person in range(9,10):
    for m_rec in range(numRec):
        # prepare data
        #X_train, X_valid, X_test, y_train, y_valid, y_test = DP.Individual_LeaveRecOut(m_test, m_rec)
        # put back to data provider later
        X_train = np.zeros((0,400,4,1))
        y_train = np.zeros((0))
        for i in range(numRec):
            if i==m_rec:
                X_test, y_test = DP.loadXY(m_person, m_rec)
            else:
                X, y = DP.loadXY(m_person, m_rec)
                X_train=np.concatenate((X_train,X),0)
                y_train=np.concatenate((y_train,y),0)
    
        #%% prepare model
        if model_type == 'TConv':
            model = MB.build_TConv(filters = 40, kernel = (40,4), dense=100,numClass = numClass)
        elif model_type == 'Cov1D':
            model = MB.build_Conv1D(filters = 40, kernel = (40), dense=100,numClass = numClass)
        elif model_type == 'ResConv1D':
            model = MB.build_ResConv1D(filters = 40, kernel = (40), dense=100,numClass = numClass)
        elif model_type == 'LSTM':
            model = MB.build_LSTM(lstm_units = 40, dense=100)
        elif model_type == 'Conv_LSTM':
            model = MB.build_Conv_LSTM(conv_filters = 20, conv_kernel = (40,4), lstm_units = 40, dense = 100, numClass = numClass)
        elif model_type == 'Conv1D_LSTM':
            model = MB.build_Conv1D_LSTM(conv_filters = 20, conv_kernel = (40), lstm_units = 40, dense = 100, numClass = numClass)
        elif model_type == 'TfEncoder':
            model = MB.build_TfEncoder(batch)
            
        #m_opt = keras.optimizers.Adam(learning_rate=0.0005)
        #m_opt = keras.optimizers.SGD(learning_rate=0.005, momentum=0.001)
        m_opt = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.001)
        model.compile(optimizer=m_opt,
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        model.summary()
        #%% train model
        acc = []
        val_acc = []
        loss = []
        val_loss = []
        epoch = 1000
        model, history = tools.train_step_open(model, epoch, X_train, y_train, 
                                          modelsavefile, batch_size = batch)
        #acc, val_acc, loss, val_loss = tools.append_history(history, acc, val_acc, loss, val_loss)
        #tools.plot_acc_loss(acc, val_acc, loss, val_loss, file_path=outfolder+'P_'+str(m_person)+'_'+model_type)
        #%% test
        #model.load_weights(modelsavefile) #model needs to be built first 
        y_predict = model.predict(X_test)
        acc_test = sum(y_test == np.argmax(y_predict, axis=1)) / y_test.shape[0]
        cm = confusion_matrix(y_test, np.argmax(y_predict, axis=1))
        tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path=outfolder+'P_'+str(m_person)+'_'+model_type,acc=acc_test)
        cm_all = np.concatenate( (cm_all, np.expand_dims(cm,2)), 2)
        
        tools.save_history(outfolder+'P_'+str(m_person)+'_'+model_type, acc, val_acc, loss, val_loss, cm)
        
        print(acc_test)
        print(cm)
#%%
cm = np.sum(cm_all,2)
acc = np.sum(cm*np.eye(numClass, numClass)) / np.sum(cm)
tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path=outfolder+'Individual_'+model_type,acc=acc)
print(acc)
print(cm)
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:46:24 2021

@author: bzhou
"""
from tensorflow import keras

import numpy as np
import toolbox as tools
import ModelBuilder as MB
import DataProvider as DP

from itertools import combinations

from sklearn.metrics import confusion_matrix

session = tools.tf_mem_patch()

outfolder = 'outputs/LPO_AS/'
model_type = 'Conv1D'  # Conv1D, TConv, LSTM, Conv_LSTM, TfEncoder, Conv1D_LSTM, ResConv1D
modelsavefile = 'model/'+model_type+'_LPO_AS.h5'
numClass = 20
m_population = 10
batch = 240
cm_all = np.zeros((numClass, numClass, 0))
logFile = tools.create_log(outfolder,['condition','testID','validID','best valid acc','best test acc'])
numValid = 1
for m_test in range(m_population):
    
    m_train = list(range(m_population))
    m_train.remove(m_test)
    valid_list = combinations(m_train, numValid)
    
    for m_valid1 in valid_list:
        # training index is population removing test subject
        m_valid1 = list(m_valid1)
        str_valid = ''
        m_train = list(range(m_population))
        m_train.remove(m_test)
        for m_id in m_valid1:
            m_train.remove(m_id)
            str_valid+=str(m_id)
        # prepare data
        X_train, X_valid, X_test, y_train, y_valid, y_test = DP.LeavePersonOut_AllStrangers(m_train,m_valid1,[m_test])
    
        #%% prepare model
        if model_type == 'TConv':
            model = MB.build_TConv(filters = 40, kernel = (40,4), dense=100)
        elif model_type == 'Conv1D':
            model = MB.build_Conv1D(filters = 40, kernel = (40), dense=100)
        elif model_type == 'ResConv1D':
            model = MB.build_ResConv1D(filters = 40, kernel = (40), dense=100)
        elif model_type == 'LSTM':
            model = MB.build_LSTM(lstm_units = 40, dense=100)
        elif model_type == 'Conv_LSTM':
            model = MB.build_Conv_LSTM(conv_filters = 20, conv_kernel = (40,4), lstm_units = 40, dense = 100, numClass = 20)
        elif model_type == 'Conv1D_LSTM':
            model = MB.build_Conv1D_LSTM(conv_filters = 20, conv_kernel = (40), lstm_units = 40, dense = 100, numClass = 20)
        elif model_type == 'TfEncoder':
            model = MB.build_TfEncoder(batch)
        
        m_opt = keras.optimizers.Adam(learning_rate=0.00005)
        model.compile(optimizer=m_opt,
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        model.summary()
        #%% train model
        acc = []
        val_acc = []
        loss = []
        val_loss = []
        patience = 500
        epoch = 10000
        model, history = tools.train_step(model, epoch, X_train, y_train, X_valid, y_valid, 
                                          modelsavefile, Patience = patience, batch_size = batch)
        acc, val_acc, loss, val_loss = tools.append_history(history, acc, val_acc, loss, val_loss)
        tools.plot_acc_loss(acc, val_acc, loss, val_loss, 
                            file_path=outfolder+'LPO_'+str(m_test)+'_'+str_valid+'_'+model_type)
        #%% test
        model.load_weights(modelsavefile) #model needs to be built first 
        y_predict = model.predict(X_test)
        acc_test = sum(y_test == np.argmax(y_predict, axis=1)) / y_test.shape[0]
        cm = confusion_matrix(y_test, np.argmax(y_predict, axis=1))
        tools.plot_confusion_matrix(cm, range(1, numClass+1), 
                                    file_path=outfolder+'LPO_'+str(m_test)+'_'+str_valid+'_'+model_type,acc=acc_test)
        cm_all = np.concatenate( (cm_all, np.expand_dims(cm,2)), 2)
        
        tools.save_history(outfolder+'LPO_'+str(m_test)+'_'+str_valid+'_'+model_type, acc, val_acc, loss, val_loss, cm)
        
        print(acc_test)
        print(cm)
        tools.write_log_line(logFile,[model_type+'LPO_'+str(m_test)+'_'+str_valid,
                                      m_test,str_valid,round(val_acc.max(),4),round(acc_test,4)])

cm = np.sum(cm_all,2)
acc = np.sum(cm*np.eye(numClass, numClass)) / np.sum(cm)
tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path=outfolder+'LPO_AllStrangers_'+model_type,acc=acc)
print(acc)
print(cm)
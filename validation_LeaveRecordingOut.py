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

outfolder = 'outputs/LRO/'
model_list = ['TConv'] #'Conv1D','TConv','ResConv1D','LSTM','Conv1D_LSTM','Conv_LSTM','TfEncoder','Conv1D_noflat','DC_LSTM'
numClass = 20
m_population = 14
batch = 400
numRec = 5

for itr in range(5):
    cm_all = np.zeros((numClass, numClass, 0))
    logFile = tools.create_log(outfolder,['condition','best valid acc','best test acc',tools.get_time_str()])
    
    
    for model_type in model_list:
        for m_rec in range(numRec):
            modelsavefile = 'model/'+model_type+'_LRO_'+str(m_rec)+'.h5'
            # leave recording out
            X_train, X_valid, X_test, y_train, y_valid, y_test = DP.Group_LeaveRecOut(list(range(m_population)), m_rec)
            
            #% prepare model
            
            if model_type == 'TConv':
                model = MB.build_TConv(filters = 40, kernel = (40,4), dense=100)
            elif model_type == 'Conv1D':
                model = MB.build_Conv1D(filters = 40, kernel = (40), dense=100)
            elif model_type == 'ResConv1D':
                model = MB.build_ResConv1D(filters = 40, kernel = (40), dense=100)
            elif model_type == 'LSTM':
                model = MB.build_LSTM(lstm_units = 64, dense=128)
            elif model_type == 'Conv_LSTM':
                model = MB.build_Conv_LSTM(conv_filters = 20, conv_kernel = (40,4), lstm_units = 40, dense = 100, numClass = 20)
            elif model_type == 'Conv1D_LSTM':
                model = MB.build_Conv1D_LSTM(conv_filters = 20, conv_kernel = (40), lstm_units = 40, dense = 100, numClass = 20)
            elif model_type == 'TfEncoder':
                model = MB.build_TfEncoder(batch)
            elif model_type == 'DC_LSTM':
                model = MB.build_DeepConvLSTM()
            elif model_type == 'Conv1D_noflat':
                model = MB.build_Conv1D_noflat(filters = 40, kernel = (40), dense=100)
                
            # optimizer
            m_opt = keras.optimizers.Adam(learning_rate=0.00005)
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
            tools.save_history(outfolder+'LeaveRecOut_'+str(m_rec)+'_'+model_type, acc, val_acc, loss, val_loss, cm)
            
            print(acc_test)
            print(cm)
            tools.write_log_line(logFile,[model_type+'_LRO_'+str(m_rec),round(val_acc.max(),4),acc_test,tools.get_time_str()])
            
        cm = np.sum(cm_all,2)
        acc = np.sum(cm*np.eye(numClass, numClass)) / np.sum(cm)
        tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path=outfolder+'Result_LRO_'+model_type,acc=acc)
        print(acc)
        print(cm)
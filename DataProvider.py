# -*- coding: utf-8 -*-
"""
Created on Fri May 21 22:37:39 2021

@author: bzhou
"""
import numpy as np
import toolbox as tools

def Individual_LeaveRecOut(m_person=1):
    numClass = 20
    X=np.load('data/P'+str(m_person)+'_X.npy')
    y=np.load('data/P'+str(m_person)+'_y.npy')
    y=y-1
    #expanding dim only needed if channels are taken into the 2D format, virtually adding 1 channel at the end of dims
    X=np.expand_dims(X,3)
    y_cat=tools.label2categorical(y,numClass) #from (windows,1) to (windows,20)
    #train-valid-test
    ind_train = range(0,300)  #consider to parameterize
    ind_valid = range(300,320)#consider to parameterize
    ind_test  = range(320,400)#consider to parameterize
    
    X_train = X[ind_train, :, :, :]
    X_valid = X[ind_valid, :, :, :]
    X_test  = X[ind_test,  :, :, :]
    
    y_train = y_cat[ind_train,:]
    y_valid = y_cat[ind_valid,:]
    y_test  = y[ind_test]
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def Group_LeaveRecOut(m_group = [1,2,3]):
    numClass = 20
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test
    
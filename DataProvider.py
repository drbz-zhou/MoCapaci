# -*- coding: utf-8 -*-
"""
Created on Fri May 21 22:37:39 2021

@author: bzhou
"""
import numpy as np
import toolbox as tools

def loadXY(m_person):
    X=np.load('data/P'+str(m_person)+'_X.npy')
    y=np.load('data/P'+str(m_person)+'_y.npy')
    y=y-1
    return X, y

def Individual_LeaveRecOut(m_person=1):
    numClass = 20
    X, y = loadXY(m_person)
    #expanding dim only needed if channels are taken into the 2D format, virtually adding 1 channel at the end of dims
    X=np.expand_dims(X,3)
    #train-valid-test
    ind_train = range(0,300)  #consider to parameterize
    ind_valid = range(300,320)#consider to parameterize
    ind_test  = range(320,400)#consider to parameterize
    
    X_train = X[ind_train, :, :, :]
    X_valid = X[ind_valid, :, :, :]
    X_test  = X[ind_test,  :, :, :]
    
    y_train = y[ind_train]
    y_valid = y[ind_valid]
    y_test  = y[ind_test]
    
    # model training requires categorical
    y_train = tools.label2categorical(y_train,numClass)
    y_valid = tools.label2categorical(y_valid,numClass)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def Group_LeaveRecOut(m_group = [1,2,3]):
    numClass = 20
    #initialize
    X_train = np.empty( (0,400,4,1) )
    X_valid = np.empty( (0,400,4,1) )
    X_test = np.empty( (0,400,4,1) )
    y_train = np.empty( (0) )
    y_valid = np.empty( (0) )
    y_test = np.empty( (0) )
    
    ind_train = range(0,300)  #consider to parameterize
    ind_valid = range(300,320)#consider to parameterize
    ind_test  = range(320,400)#consider to parameterize
    for m_person in m_group:
        X, y = loadXY(m_person)
        X=np.expand_dims(X,3)
        X_train = np.concatenate( (X_train, X[ind_train, :, :, :]) )
        X_valid = np.concatenate( (X_valid, X[ind_valid, :, :, :]) )
        X_test  = np.concatenate( (X_test,  X[ind_test,  :, :, :]) )
        y_train = np.concatenate( (y_train, y[ind_train]) )
        y_valid = np.concatenate( (y_valid, y[ind_valid]) )
        y_test  = np.concatenate( (y_test , y[ind_test]) )
    
    # model training requires categorical
    y_train = tools.label2categorical(y_train,numClass)
    y_valid = tools.label2categorical(y_valid,numClass)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def LeavePersonOut_Stranger(m_trainGroup=[1,2,3], m_testGroup=[4,5]):
    numClass = 20
    #initialize
    X_train = np.empty( (0,400,4,1) )
    X_valid = np.empty( (0,400,4,1) )
    X_test = np.empty( (0,400,4,1) )
    y_train = np.empty( (0) )
    y_valid = np.empty( (0) )
    y_test = np.empty( (0) )
    # last recording per person as validation
    ind_train = range(0,400)
    ind_valid = range(320,400)
    for m_person in m_trainGroup:
        X, y = loadXY(m_person)
        X=np.expand_dims(X,3)
        X_train = np.concatenate( (X_train, X[ind_train, :, :, :]) )
        X_valid = np.concatenate( (X_valid, X[ind_valid, :, :, :]) )
        y_train = np.concatenate( (y_train, y[ind_train]) )
        y_valid = np.concatenate( (y_valid, y[ind_valid]) )
    for m_person in m_testGroup:
        X, y = loadXY(m_person)
        X=np.expand_dims(X,3)
        X_test  = np.concatenate( (X_test,  X) )
        y_test  = np.concatenate( (y_test , y) )
    
    # model training requires categorical
    y_train = tools.label2categorical(y_train,numClass)
    y_valid = tools.label2categorical(y_valid,numClass)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def LeavePersonOut_Acquaintance(m_trainGroup=[1,2,3], m_testGroup=[4,5]):
    numClass = 20
    #initialize
    X_train = np.empty( (0,400,4,1) )
    X_valid = np.empty( (0,400,4,1) )
    X_test = np.empty( (0,400,4,1) )
    y_train = np.empty( (0) )
    y_valid = np.empty( (0) )
    y_test = np.empty( (0) )
    # last recording per person as validation
    ind_valid = range(0,80)
    ind_test  = range(80,400)
    for m_person in m_trainGroup:
        X, y = loadXY(m_person)
        X=np.expand_dims(X,3)
        X_train = np.concatenate( (X_train, X) )
        y_train = np.concatenate( (y_train, y) )
    for m_person in m_testGroup:
        X, y = loadXY(m_person)
        X=np.expand_dims(X,3)
        X_valid = np.concatenate( (X_valid, X[ind_valid, :, :, :]) )
        X_test  = np.concatenate( (X_test,  X[ind_test,  :, :, :]) )
        y_valid = np.concatenate( (y_valid, y[ind_valid]) )
        y_test  = np.concatenate( (y_test , y[ind_test]) )
    
    # model training requires categorical
    y_train = tools.label2categorical(y_train,numClass)
    y_valid = tools.label2categorical(y_valid,numClass)
    return X_train, X_valid, X_test, y_train, y_valid, y_test
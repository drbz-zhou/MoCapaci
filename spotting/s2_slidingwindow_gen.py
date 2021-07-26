# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:51:55 2021

@author: bzhou
"""
# normalize full stream
# sliding window to generate X, y
# >75% of 1-20
# win size = 400
# win step = 100
import numpy as np
import matplotlib.pyplot as plt

for m_rec in range(0,5):  # 0,5
    for m_Person in [13]:
        data = np.load("DataExp/P" + str(m_Person) + "_" + str(m_rec) + "_np.npy")
        label = np.load("Labels/P" + str(m_Person) + "_" + str(m_rec) + "_L_np.npy")
        
        #normalize
        data_norm = np.zeros( (len(data), 4) )
        for ch in range(4):
            data_norm[:,ch] = (data[:, ch]-np.mean(data[:,ch]) ) / np.std(data[:,ch])
        #plt.plot(data_norm)
        #plt.show()
        
        X = data_norm
        y = np.zeros(len(X))
        for i in range(0, len(label)):
            for j in range(label[i,0],label[i,1]):
                y[j]=label[i,2]
        #plt.plot(y)
        #plt.show()
        
        np.save("data/P" + str(m_Person) + "_" + str(m_rec) + "_X", X)
        np.save("data/P" + str(m_Person) + "_" + str(m_rec) + "_y", y)
        
        # without 1 second normalize
        '''
        X_wo_norm = (data-np.mean(data[0:100,:],0))/20
        np.save("data_wo_norm/P" + str(m_Person) + "_" + str(m_rec) + "_X", X_wo_norm)
        np.save("data_wo_norm/P" + str(m_Person) + "_" + str(m_rec) + "_y", y)
        plt.plot(X_wo_norm)
        plt.show()
        '''
        # sliding window
        '''
        win_ratio = 0.75
        win_size = 400
        win_step = 100
        win_amount = int( (len(X)-win_size)/win_step )
        X_sw = []
        y_sw = []
        for i in range(0, win_amount):
            w_start = i*win_step
            w_end   = w_start + win_size
            X_sw.append(X[w_start:w_end,:])
            u,v = np.unique(y[w_start:w_end], return_counts=True)
            y_temp = 0
            if np.max(v) > win_ratio*win_size:  # if majority > win_ratio of window
                y_temp = int(u[np.argmax(v)])
            y_sw.append(y_temp)
        
        #np.save("data/P" + str(m_Person) + "_" + str(m_rec) + "_X_sw", X_sw)
        #np.save("data/P" + str(m_Person) + "_" + str(m_rec) + "_y_sw", y_sw)
        
        #plt.plot(y_sw)
        #plt.show()
        '''
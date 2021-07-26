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
    for m_Person in [9]:
        data = np.load("DataExp/P" + str(m_Person) + "_" + str(m_rec) + "_np.npy")
        label = np.load("Labels/P" + str(m_Person) + "_" + str(m_rec) + "_L_np.npy")
        
        #normalize
        data_norm = np.zeros( (len(data), 4) )
        for ch in range(4):
            data_norm[:,ch] = (data[:, ch]-np.mean(data[:,ch]) ) / np.std(data[:,ch])
        plt.plot(data_norm)
        plt.show()
        
        X = data_norm
        y = np.zeros(len(X))
        for i in range(0, len(label)):
            for j in range(label[i,0],label[i,1]):
                y[j]=label[i,2]
        plt.plot(y)
        plt.show()
        
        np.save("TrainingData/P" + str(m_Person) + "_" + str(m_rec) + "_X", X)
        np.save("TrainingData/P" + str(m_Person) + "_" + str(m_rec) + "_y", y)
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:44:43 2021

@author: bzhou
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for m_rec in range(3,4):
        for m_Person in [9]:
            data = np.load("DataExp/P" + str(m_Person) + "_" + str(m_rec) + "_np.npy")
            label = np.load("Labels/P" + str(m_Person) + "_" + str(m_rec) + "_L_np.npy")
            plt.plot(data)
            plt.plot(np.mean(data,1))
            plt.show()
            #%%
            m_roll_win = 60000
            m_roll_half = int(m_roll_win/2)
            data_padded = np.zeros((len(data)+m_roll_win, 4) )
            data_padded[m_roll_half:len(data)+m_roll_half,:]=data
            for i in range(0,m_roll_half):
                data_padded[i,:] = data[0,:]
                data_padded[-i,:] = data[-1,:]
            m_drift = np.zeros((len(data),4))
            for i in range(0, len(m_drift)):
                m_drift[i,:] = np.mean(data[i:i+m_roll_win,:],0)
            plt.plot(m_drift)
            plt.show()
            
            #%%
            plt.figure()
            plt.plot(data-m_drift)
            plt.show()
            #%%
            # rules of sliding window
            # win size > np.max(label[:,1]-label[:,0])
            # win step 1 - around 50, win step 2 - after previous confirmed window
            # all parameters should be from the training data, e.g. 0-test, 1,2,3,4-training, 
            # but the model can be trained by the rest people, and fine tuned by the test person
            # rule 1: if std(win) < TH_1, consider as null - need to check the TH_1
            # rule 2: if abs(end - first) > Th_2, remove, continue with win step 1
            # rule 3: else: shrink window, with a, c, d, b to capture the actual activity:
            #   e.g. while (c=a+1) until abs(win(c)-win(a)) > TH_1, or std(win(a:c)) > TH_1
            #        same for d->b, and consider as activity
            m_win_size = 700  #calculate for P9
            m_win_step = 50 
            channel_range = 1 # 0-4
            ##To calculate
            std_th_1 = 0
            null_windows = list()
            valid_windows = list()
            label_first_null = list()
            label_end_null = list()
            label_first_valid = list()
            label_end_valid = list()
            ##To calculate
            std_th_2 = 0
            
            
         
            if(m_win_size > np.max(label[:,1]-label[:,0])): # ok 
             
                for i in range(len(data[0])):
                        temp_window = data[y][label[i,0] : label[i + m_win_size,1]]
                        first = data[y][label[i,0]]
                        end = data[y][label[i + m_win_size,1]]
                        
                        if(abs(np.std(temp_window)) < std_th_1):
                            null_windows.append(temp_window)
                            label_first_null.append(first) 
                            label_end_null.append(end)
                        else if (abs(end-first) > std_th_2 ):
                            i = i + m_win_step
                        else: 
                            #Shrink
                            a = label[i,0]
                            c = a
                            b = label[i + m_win_size,1]
                            d = b
                            window_a = data[y][a]
                            window_c = data[y][a]
                            window_b = data[y][b]
                            window_d = data[y][b]
                            while(abs(window_c - window_a) < std_th_1):
                                c = c + 1
                            c = c - 10 # going back a little
                            while(abs(window_b - window_d) < std_th_1): 
                                d = d - 1
                            d = d + 10 # going forward a little
                            w_valid = data[y][c:d] 
                            valid_windows.append(w_valid)
                            label_first_valid.append(c)
                            label_end_valid.append(d)
                            
                            i = i + b  ## To start again the sliding after the act was capture
                            
                            
                        
                        
    
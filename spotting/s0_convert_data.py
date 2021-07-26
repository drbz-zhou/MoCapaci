import pandas as pd
import numpy as np
import math

####################################################################################################################
'''Convert from panda to numpy with only the things we need for spotting'''
####################################################################################################################


if __name__ == '__main__':
    for m_rec in range(0,5):
        for m_Person in range(9):
            # read data 
            file_data = "DataExp/P" + str(m_Person) + "_" + str(m_rec) + ".csv"
            data_1 = pd.read_csv(file_data, delimiter=',',
                               names=['Time1', 'Id1', 'Pitch1', 'Volume1', 'Time2', 'Id2', 'Pitch2', 'Volume2',
                                      'FigCounter',
                                      'TryCounter', 'greenL', 'redL', 'yellowL', 'synch'],
    
                               dtype={'Time1': int, 'Id1': str, 'Pitch1': int, 'Volume1': int, 'Time2': int,
                                      'Id2': str, 'Pitch2': int,
                                      'Volume2': int, 'FigCounter': int, 'TryCounter': int, 'greenL': bool,
                                      'redL': bool, 'yellowL': bool, 'synch': int}
                               )
            data_f = data_1[['Pitch1','Volume1','Pitch2','Volume2']]
            data_np = data_f.to_numpy()
            
            np.save("DataExp/" + str(m_Person) + "_" + str(m_rec) + "_np", data_np)
            # read label
            file_label = "Labels/P" + str(m_Person) + "_" + str(m_rec) + "_L.csv"
            labels = pd.read_csv(file_label, delimiter=',',
                         names=['nothing', 'go_g', 'middle_minus', 'middle', 'goback_g', 'middle_plus', 'relax_g', 'fig_id',
                                'try_id'],
                         dtype={'nothing': int, 'go_g': int, 'middle_minus': int, 'middle': int, 'middle_plus': int,
                                'goback_g': int, 'relax_g': int, 'fig_id': int, 'try_id': int}
                         )
            label_f = labels[['go_g','relax_g','fig_id']]
            label_np = label_f.to_numpy()
     
            np.save("Labels/" + str(m_Person) + "_" + str(m_rec) + "_L_np", label_np)
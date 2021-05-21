import pandas as pd
import numpy as np

####################################################################################################################
'''Merged Data File'''
####################################################################################################################
Volunteer_name = "luis"
hp_folder = "AfterFilter/"
all_together_file = hp_folder + Volunteer_name + "AfterFilter_All" + ".csv"

data_filter_all = pd.read_csv(all_together_file, delimiter=',',
                              names=['nothing', 'filter_pitch1', 'filter_volume1', 'filter_pitch2', 'filter_volume2',
                                     'Activity_id', 'try_id'],

                              dtype={'filter_pitch1': float, 'filter_volume1': float, 'filter_pitch2': float,
                                     'filter_volume2': float,
                                     'Activity_id': int, 'try_id': int
                                     }
                              )

####################################################################################################################
'''Output npy files'''
####################################################################################################################
Volunteer_name = "luis"
npy_folder = "ToNumpy/"
X_together_numpy = npy_folder + Volunteer_name + "_X_ToNumpy_All" + ".npy"
Y_together_numpy = npy_folder + Volunteer_name + "_Y_ToNumpy_All" + ".npy"

repetitions = 80*5
windows_data = 400
channels = 4
array_data = np.zeros((repetitions, windows_data, channels)) # numpy array (80x5,timesteps,channel)

data_filter_all_labels = data_filter_all.drop(columns=['nothing', 'filter_pitch1', 'filter_volume1', 'filter_pitch2', 'filter_volume2',
                                     'try_id'])

data_filter_all_labels = data_filter_all_labels.to_numpy()

########################################################
''' YYYYYYYYYYYYYYYYYYYY'''
########################################################

array_label = np.zeros(repetitions,dtype=int)
for counter in range(0,repetitions):
    array_label[counter] = data_filter_all_labels[counter*windows_data]

print(len(array_label))
print(array_label)
np.save(Y_together_numpy,array_label)

########################################################
''' XXXXXXXXXXXXXXXXXXXX'''
########################################################
z = 0
for y in range(0,repetitions):
    z = 0
    for i in range(y*windows_data,windows_data*(y+1)):
        array_data[y][z][0] = data_filter_all['filter_pitch1'][i]
        array_data[y][z][1] = data_filter_all['filter_volume1'][i]
        array_data[y][z][2] = data_filter_all['filter_pitch2'][i]
        array_data[y][z][3] = data_filter_all['filter_volume2'][i]
        z = z + 1
z = 0

print(len(array_data[0]))
print(array_data.shape)
np.save(X_together_numpy,array_data)






# Example of application of CNN that correlates the probability of obtaining a certain stress tensor (6 outputs)
# given a succession of N strain tensors (Nx6 inputs).

# Loading libraries, tensorflow need to be available (loaded, in the environment)
## Generic stuff...
import numpy as np
from pylab import *
import seaborn as sns
from random import randint
import pandas as pd
from scipy.spatial import ConvexHull
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
## Related to NNs
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Deconvolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Conv1D, multiply
from keras.layers.convolutional import ZeroPadding2D
from sklearn.preprocessing import StandardScaler

# Creating a function to parse the datafiles output by processes of the simulation, which contain.
# Each file contains the data of a bunch of QPs (gathered by the id of the processor that was dealing
# with the QP at the time of the FE simulation). The function organises the data in input (x) and output (y)
# features for the ML model.
def load_results_file(number_file):
    path = '/lustre/home/uccamva/tmp/interpol_stress_from_strain/pure_epoxy_cyclic_microcube/'
    file_name = path + 'pr_' + str(number_file) +'.lhistory.csv'
    data_pd = pd.read_csv(file_name)
    data_list = data_pd.values
    # insert first column with unique cell/qp ID (cqid): cell*8+q
    data_list = np.insert(data_list, 0, data_list[:,1]*8+data_list[:,2], axis=1)
    # sort by cqid then timestep
    data_list = data_list[np.lexsort((data_list[:, 1], data_list[:, 0]))]
    # Building empty containers for the current input (x) and output (y) data
    num_timestep = np.max(data_list[:,1])
    num_qps = np.size(np.unique(data_list[:,0]))
    samples_data_x = np.empty([num_qps,num_timestep*6])
    samples_data_y = np.empty([num_qps,6])
    # Creating a 2D-array (samples_data_x) where each row 'R' is the strain trajectory of a quadrature point of the following shape:
    # [qpR.time1.strain1,...,qpR.time1.strain6,qpR.time2.strain1,...,qpR.time2.strain6,...,qpR.timeT.strain1,...,qpR.timeT.strain6]
    # the number of rows is the number of QPs (num_qps), and note that 'T' is the number of timesteps (num_timestep)
    cqid = -1
    nq = 0
    for j in range(np.size(data_list[:,0])):
        if data_list[j,0] != cqid:
            if j > 0:
                samples_data_x[nq,:] = qp_strain_traj
                nq+=1
            cqid = data_list[j,0]
            qp_strain_traj = data_list[j,5:11]
        else:
            qp_strain_traj = np.append(qp_strain_traj,data_list[j,5:11])
    samples_data_x[nq,:] = qp_strain_traj

    ### [SPLINE] APPLY SPLINE-BASED DIMENSION REDUCTION HERE ###
    # Use spline fitting on the strain trajectories in order to reduce the length of the rows of "samples_data_x"
    # (i.e. the number of columns from (T*6) to (N*6) where "T" (the total number of timesteps) to "N" (the number
    # of spline control points).
    ######

    # Repeating the procedure to build a 2D-array (samples_data_y), except that each row of the array only contains the stress
    # tensor at the last timestep of a quadrature point, that is:
    # [qpR.timeT.stress1,...,qpR.timeT.stress6]
    cqid = -1
    nq = 0
    for j in range(np.size(data_list[:,0])):
        if data_list[j,0] != cqid:
            if j > 0:
                samples_data_y[nq,:] = qp_stress_last
                nq+=1
            cqid = data_list[j,0]
            qp_stress_last = data_list[j,-6:]
        else:
            qp_stress_last = data_list[j,-6:]
    samples_data_y[nq,:] = qp_stress_last
    return samples_data_x, samples_data_y

# Parameters
file_num = 10

# Generate training data
for i in range(file_num):
    # Reading data, and parsing it "slightly", from the data file produced by one process during the simulation
    samples_data_x_file, samples_data_y_file = load_results_file(i)
    # Assembling the datasets for the ML model
    if i == 0:
        ### loading strain data (N steps of 6-components) into train_X for each qp of each cell at every timestep
    	train_X = samples_data_x_file
        ### loading axial stress data into train_y for the corresponding qp of each cell at every timestep
    	train_y = samples_data_y_file
    else:
        ### continuing identlically to load the data for the subsequent files
    	train_X = np.append(train_X,samples_data_x_file,axis=0)
    	train_y = np.append(train_y,samples_data_y_file,axis=0)

print('train X shape: ',train_X.shape)
print('train y shape: ',train_y.shape)

### using the the "num_file+1" data file as the testing data
test_X, test_y = load_results_file(i+1)

print('test X shape: ',test_X.shape)
print('test y shape: ',test_y.shape)

### Normalization of the features data set considering that each line
### ([strain_11,strain_12,strain_13,strain_22,strain_23,strain_33]) is an (independent?) observation
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)
### Do we not need to normalise the output data?


################################ Model and Training ####################################
### What type of algorithm/model is a "Sequential"?
model = Sequential()
model.add(Dense(6,activation='relu',input_shape=(np.size(train_X[0,:]),)))
model.add(Dense(12, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(np.size(train_y[0,:]),activation='linear'))

model.compile(loss='mse',optimizer='adam')
model.summary()

### What is the criteria to choose the number of epochs and the batch size?
### What do they actually mean?
model.fit(train_X,train_y,epochs=100, batch_size=100)

# ################################ Testing ##############################################
y_pred = model.predict(test_X)
plt.plot(y_pred[:,0],'r*',label='prection via dl')
plt.plot(test_y[:,0],'b*',alpha=0.5,label='Numerical result')
plt.legend()
plt.show()

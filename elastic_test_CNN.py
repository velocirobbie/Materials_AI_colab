# Example of application of NN that correlates the probability of obtaining a certain axial stress (1 output)
# given the current strain tensor (6 inputs).

# General questions/issues:
## - Can we execute this in parallel? On cpus, on gpus? Easy to switch!
## - When we introduce time-dependency the feature list will have an higher dimension. In order to have
##   a constant set of features, which the order matters but not the exact time, we might have to use
##   splines with a fixed number of control points as well
#### - When we use splines to limit the number of strain tensors required to describe the strain history,
####   what we have is a high-dimensional set of inputs equal to the number of control points in the spline
####   times the number of independent components in the strain tensor (ncp*6)
#### - If we consider that each new MD simulation is a new sample (of the dataset on which we train the surrogate model)
####   we need a model that can be trained on a growing dataset (for which the inputs dimension remains constant)
## - Another possibility is to consider that the inputs dimension is variable (each sample of the dataset can have a variable
##   amount of inputs), is there machine learning algorithms that can be trained on a dataset of samples which do not have the sample
##   inputs dimension (does it make any sense)? Besides, it would also have to be trained on a growing dataset.
## - It also apparently possible to reduce the dimension on the features using PCA

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

# Creating a function to parse the datafiles, which contain all the quadrature points information
# during the simulation. Each file contains the data of a bunch of QPs (gathered by the id of
# the processor that was dealing with the QP at the time of the FE simulation)
def load_results(number_file):
    path = '/lustre/home/uccamva/tmp/interpol_stress_from_strain/pure_epoxy_cyclic_microcube/'
    file_name = path + 'pr_' + str(number_file) +'.lhistory.csv'
    data_pd = pd.read_csv(file_name)
    data_raw =data_pd.values
    return data_raw



#######################  Clean data ############################
# Get data at time step (0,30)
# Elastic mapping test

# Parameter
file_num = 100
min_timestep = 1
max_timestep = 7
Q_point = 8
cell_file = 10
num_history = 6

### Apparently not in use...
size_train = 500
test_size = size_train/10

# Generate training points
for i in range(file_num):
	data_list = load_results(i)
	if i == 0:
        ### loading strain data (6-components) into train_X for each qp of each cell at every timestep
		train_X = data_list[:(max_timestep*Q_point*cell_file),4:10]
        ### loading axial stress data into train_y for the corresponding qp of each cell at every timestep
		train_y = data_list[:(max_timestep*Q_point*cell_file),-6]
	else:
        ### continuing identlically to load the data for the subsequent files
		train_X = np.append(train_X,data_list[:(max_timestep*Q_point*cell_file),4:10],axis=0)
		train_y = np.append(train_y,data_list[:(max_timestep*Q_point*cell_file),-6])

print('train X shape: ',train_X.shape)
print('train y shape: ',train_y.shape)

### using the the "num_file+1" data file as the testing data
test_list = load_results(file_num+1)
test_X = test_list[:(max_timestep*Q_point*cell_file),4:10]
test_y = test_list[:(max_timestep*Q_point*cell_file),-6]

print('test X shape: ',test_X.shape)
print('test y shape: ',test_y.shape)

# print(train_X[-1,:])


### Normalization of the features data set considering that each line
### ([strain_11,strain_12,strain_13,strain_22,strain_23,strain_33]) is an (independent?) observation
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)
### Do we not need to normalise the output data?


################################ Model and Training ####################################
### What type of algorithm/model is a "Sequential"?
model = Sequential()
model.add(Dense(6,activation='relu',input_shape=(6,)))
model.add(Dense(12, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mse',optimizer='adam')
model.summary()

### What is the criteria to choose the number of epochs and the batch size?
### What do they actually mean?
model.fit(train_X,train_y,epochs=100, batch_size=100)


# ################################ Testing ##############################################
y_pred = model.predict(test_X)
plt.plot(y_pred,'r*',label='prection via dl')
plt.plot(test_y,'b*',alpha=0.5,label='Numerical result')
plt.legend()
plt.show()

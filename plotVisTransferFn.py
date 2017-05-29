# Plot transfer function (LGN - V1)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from HSM import HSM
from visualization import *
import my_callbacks

import time

download_time = time.time()
#Keras with tensorflow backend
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.optimizers import SGD

def get_trained_Ks(Ks, num_LGN=9):
#Note : add num_lgn and hlsr later  DEFAULT num_LGN = 9 , hlsr = 0.2 --> layers 
# x,y = center coordinate
	n = num_LGN
	x=Ks[0:n]; 
	i=1; y=Ks[n*i:n*(i+1)]; 
	i=2; sc=Ks[n*i:n*(i+1)]; i=3; ss=Ks[n*i:n*(i+1)]; i=4; rc=Ks[n*i:n*(i+1)]; i=5; rs=Ks[n*i:n*(i+1)];
	return x,y,sc,ss,rc,rs


def get_LGN_out(X,x,y,sc,ss,rc,rs):
	# X = input image 
	img_vec_size=int(np.shape(X)[0])
	img_size = int(np.sqrt(img_vec_size ))
	num_LGN= np.shape(sc)[0]
	
	xx,yy = np.meshgrid(np.arange(img_size),np.arange(img_size))
	xx = np.reshape(xx,[img_vec_size]); yy = np.reshape(yy,[img_vec_size]);
	
	#Below
	lgn_kernel = lambda i,x,y,sc,ss,rc,rs: np.dot(X, ((rc[i]*(np.exp(-((xx- x[i])**2 + (yy - y[i])**2)/2/sc[i]).T/(2*sc[i]*np.pi))) - rs[i]*(np.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2 /(sc[i]+ss[i])).T/(2*(sc[i]+ss[i])*np.pi))))
	
	lgn_ker_out = np.ndarray([num_LGN],dtype=float)
	
	for i in np.arange(num_LGN):
		lgn_ker_out[i] = lgn_kernel(i,x,y,sc,ss,rc,rs)
	
	return lgn_ker_out

def cmpr_hist_pred_record(P, Y, num_bins = 20):
	
	input_shape = np.shape(P)
	PP = np.reshape(P,[input_shape[0]*input_shape[1],1])
	YY = np.reshape(Y,[input_shape[0]*input_shape[1],1])
	
	# Two subplots, unpack the axes array immediately
	f, (ax1, ax2) = plt.subplots(1, 2)
	ax1.hist(YY,num_bins)
	ax1.set_title('Recorded neural response')
	ax2.hist(PP,num_bins)
	ax2.set_title('Predicted response')
	plt.show()

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
######################################################
## MAIN CODE
######################################################

# DEFAULT num_LGN = 9 , hlsr = 0.2 --> The hidden layer size ratio
NUM_LGN = 9; HLSR = 0.2 

# fix random seed for reproducibility
np.random.seed(123)


#load data
train_input=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_inputs.npy') # 1800, 961
train_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_set.npy') #1800, 103
#load validation data
raw_vld_set = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/raw_validation_set.npy')
vldinput_set = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/validation_inputs.npy')
vld_set = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/validation_set.npy')

#size 
img_vec_size =int(np.shape(train_input)[1])
img_size = int(np.sqrt(img_vec_size ))


#load trained LGN hyperparameters
num_LGN = NUM_LGN; hlsr = HLSR;
[Ks,hsm] = np.load('out_region1.npy')

#load trained parameters for DoG
x,y,sc,ss,rc,rs = get_trained_Ks(Ks,9)


# num LGN = 9 L1 = 20  L2 = 103
# per image
X=np.ndarray([1800,9]) #LGN output = input to NN
for ii in np.arange(np.shape(train_input)[0]):
	lgn_ker_out = get_LGN_out(train_input[ii,:],x,y,sc,ss,rc,rs) # Output of LGN = input to NN
	X[ii,:]=lgn_ker_out #Output from LGN

Y = train_set.copy() #1 All image = 1800,103

#manipulate test set
testsize=np.shape(vldinput_set)[0] # Input to NN for validation
X_test=np.ndarray([testsize,num_LGN])
for ii in np.arange(testsize):
	lgn_ker_out = get_LGN_out(vldinput_set[ii,:],x,y,sc,ss,rc,rs) # Output of LGN = input to NN
	X_test[ii,:]=lgn_ker_out #Output from LGN
Y_test = vld_set.copy()

#import pdb; pdb.set_trace()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from HSM import HSM
from visualization import *
from scipy.optimize import fmin_tnc
import param


def hist_of_pred_and_record_response(pred_response, recorded_response, cell_id=0):
  plt.subplot(121); plt.hist(recorded_response[:,cell_id]); plt.title('Recorded Response');
  plt.subplot(122); plt.hist(pred_response[:,cell_id]); plt.title('Predicted Response');
  plt.suptitle("Distribution of cell #%g's response"%cell_id)
  plt.show()

def plot_act_of_max_min_corr(yhat,train_set,corr):
    imax = np.argmax(corr) # note : actually have to combine neurons in all regions

    plt.plot(train_set[:,imax],'-ok')
    plt.plot(yhat[:,imax],'--or')
    plt.title('Cell#%d has max corr of %f'%(imax+1,np.max(corr)))
    plt.show()

    imin = np.argmin(corr) # note : actually have to combine neurons in all regions

    plt.plot(train_set[:,imin],'-ok')
    plt.plot(yhat[:,imin],'--or')
    plt.title('Cell#%d has min corr of %f'%(imin+1,np.min(corr)))
    plt.show()


download_time = time.time() #print "Download complete: Time %s" %(time.time() - download_time)
training_inputs=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_inputs.npy')
training_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_set.npy')
train_input1 = training_inputs.copy()
train_set1=training_set.copy()


raw_vld_set1 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/raw_validation_set.npy')
vldinput_set1 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/validation_inputs.npy')
vld_set1 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/validation_set.npy')


#out = fitHSM(r1_train_input,r1_train_set)



seed=13;lgn=9; hlsr=0.2

"""
This function performs fitting of the HSM model using the fmin_tnc np method.

training_inputs : 2D ndarray of inputs of shape (num of training presentations,number of pixels)
training_set    : 2D ndarray of neural responses to corresponding inputs of shape (num of training presentations,number of recorded neurons)
"""
num_pres,num_neurons = np.shape(training_set)
import ipdb; ipdb.set_trace()
print "Creating HSM model"
hsm = HSM(training_inputs,training_set) # Initialize model --> add input and output, construct parameters , build mobel, # create loss function
print "Created HSM model"   
hsm.num_lgn = lgn 
hsm.hlsr = hlsr
    
func = hsm.func() 

Ks = hsm.create_random_parametrization(seed) # set initial random values of the model parameter vector

#(Ks,success,c)=fmin_tnc(func ,Ks,fprime=hsm.der(),bounds=hsm.bounsd,maxfun = 100000,messages=0)  # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_tnc.html
(Ks,success,c)=fmin_tnc(func ,Ks,fprime=hsm.der(),bounds=hsm.bounds,maxfun = 100000,disp=5)
# Disp NIT NF F GTG
import ipdb; ipdb.set_trace()

response = HSM.response(hsm,training_inputs,Ks) # predicted response after train

corr = computeCorr(response, training_set)
plot_act_of_max_min_corr(response,training_set,corr)
hist_of_pred_and_record_response(response,training_set,cell_id=np.argmax(corr))
""" 
    	
	func : callable func(x, *args)
		Function to minimize. Must do one of:
			Return f and g, where f is the value of the function and g its gradient (a list of floats).
			Return the function value but supply gradient function separately as fprime.
			Return the function value and set approx_grad=True.
		If the function returns None, the minimization is aborted.
	x0 : array_like
		Initial estimate of minimum.
	fprime : callable fprime(x, *args)
		Gradient of func. If None, then either func must return the function value and the gradient (f,g = func(x, *args)) or approx_grad must be True
    bounds : list
		(min, max) pairs for each element in x0, defining the bounds on that parameter. Use None or +/-inf for one of min or max when there is no bound in that direction.

    maxfun : int
		Maximum number of function evaluation. if None, maxfun is set to max(100, 10*len(x0)). Defaults to None.
	
	messages :
		Bit mask used to select messages display during minimization values defined in the MSGS dict. Defaults to MGS_ALL.
		
	Returns:	
		x : ndarray
			The solution.
		nfeval : int
			The number of function evaluations.
		rc : int
			Return code as defined in the RCSTRINGS dict.

Minimize a function with variables subject to bounds, 
using gradient information in a truncated Newton algorithm. 
This method wraps a C implementation of the algorithm.
"""
print 'Final training error: ', func(np.array(Ks))/num_neurons/len(training_set)

import ipdb; ipdb.set_trace()
[Ks1,hsm1] = np.load('out_region1.npy')



print "Download complete: Time %s" %(time.time() - download_time)
import ipdb; ipdb.set_trace()
response1 = HSM.response(hsm1,train_input1,Ks1) # predicted response after train

#Validation set
pred_response1 = HSM.response(hsm1,vldinput_set1,Ks1) #predicted response for validation set

# fig 4A response vs image
# HAve to find the best neuron first
import visualization


corr1 = computeCorr(response1, train_set1)
plot_act_of_max_min_corr(response1,train_set1,corr1)
hist_of_pred_and_record_response(response1,train_set1,cell_id=np.argmax(corr1))


vld_corr1 = computeCorr(pred_response1,vld_set1)
plot_act_of_max_min_corr(pred_response1,vld_set1,vld_corr1)
hist_of_pred_and_record_response(pred_response1,vld_set1,cell_id=np.argmax(vld_corr1 ))


"""
Each directory contains the recorded data and inputs for one of the three recorded regions.
The responses are already pre-processed to give one value per image presentation (see methodology).
Each file contains one np array (saved with np.save function).
The name of the files are self-explanatory.

"""

"""
 This way, for each imaged region, we obtained two datasets of values.
 1)  The first is an n x m matrix
			- corresponding to the responses of each of the m recorded neurons to n single trial image presentations, which we refer to as the training set
 2)   Additionally, in each region
			- we recorded responses to 8-12 presentations of another 50 images forming the second dataset,
			a 50 x m x r matrix, we will refer to as the validation set.
3) Three regions in two animals were recorded,
			- containing 103, 55 and 102 neurons, while presenting sequences of 1800, 1260 and 1800 single trial images, respectively.
			-  R1: 1800(trial) x 103(neurons)  ,
			-  R2:  1260 x 55 ,
			-  R3: 1800 x 102
4)   The images were presented in partially interleaved manner.
		- The training images were divided into 10 blocks.
5)   Additional blocks were formed by the 50 validation images,
		-  in each of these blocks the 50 images were presented multiple times.
		-  During the experiment the resulting stimulation blocks were presented in random order.
6)  	For each region,
		-  we ran a rLN fitting protocol with full-field stimuli to determine the rough position and size of all the neurons' RFs.
			- Consistent with retinotopic map in mouse V1, in all three recorded regions all recovered RFs were located in a restricted region of visual space.
			This allowed us to determine a region of interest in the visual space, centered on the set of initially recovered RFs and spanning roughly two times the area they covered.
			-  The images were constrained to this region of interest and then down-sampled to 31 x 31 pixels to form the input stimuli set, which was used in all the subsequent analysis.

"""



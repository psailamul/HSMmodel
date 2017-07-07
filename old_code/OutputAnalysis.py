# check training output from Allen brain data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from HSM import HSM
from visualization import *

BIG =False
if BIG:
	#import ipdb; ipdb.set_trace()
	download_time = time.time() #print "Download complete: Time %s" %(time.time() - download_time)
	train_input=np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_train_input_big.npy')
	train_set=np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_train_output_big.npy')

	raw_vld_set1 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/raw_validation_set.npy')
	vldinput_set = np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_test_input_big.npy')
	vld_set = np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_test_output_big.npy')
	[Ks,hsm] = np.load('out_Allen_big.npy')
else:
	#import ipdb; ipdb.set_trace()
	download_time = time.time() #print "Download complete: Time %s" %(time.time() - download_time)
	train_input=np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_train_input.npy')
	train_set=np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_train_output.npy')

	raw_vld_set1 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/raw_validation_set.npy')
	vldinput_set = np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_test_input.npy')
	vld_set = np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_test_output.npy')
	[Ks,hsm] = np.load('out_Allen_unique_MSE.npy')



print "Download complete: Time %s" %(time.time() - download_time)


# predicted response after train
count_time = time.time() 
response = HSM.response(hsm,train_input,Ks) 
print "Predicted response for train set complete: Time %s" %(time.time() - count_time)

#Validation set
count_time = time.time() 
pred_response = HSM.response(hsm,vldinput_set,Ks) #predicted response for validation set
print "Predicted response for test set complete: Time %s" %(time.time() - count_time)

# fig 4A response vs image
# HAve to find the best neuron first



corr = computeCorr(response, train_set)
print "Correlation between predicted response and recorded response(Training Set): %s" %(corr)


vld_corr = computeCorr(pred_response,vld_set)
print "Correlation between predicted response and recorded response(Test Set): %s" %(corr)


[train_c, val_c] = printCorrelationAnalysis(train_set,vld_set,response,pred_response)

# Max corr neurons
imax = np.argmax(vld_corr) # note : actually have to combine neurons in all regions

plt.plot(vld_set[:,imax],'-ok')
plt.plot(pred_response[:,imax],'--or')
plt.title('Cell#%d has max corr of %f'%(imax+1,np.max(vld_corr)))
plt.show()

imin = np.argmin(vld_corr) # note : actually have to combine neurons in all regions

plt.plot(vld_set[:,imin],'-ok')
plt.plot(pred_response[:,imin],'--or')
plt.title('Cell#%d has min corr of %f'%(imin+1,np.min(vld_corr)))
plt.show()

"""
import matplotlib.image as mpimg





img = np.reshape(train_input1[1],[31,31])
plt.imshow(img, cmap="gray")
#plt.show()
"""

#img= train_input1[]
#imgplot = plt.imshow(img)

"""
Each directory contains the recorded data and inputs for one of the three recorded regions.
The responses are already pre-processed to give one value per image presentation (see methodology).
Each file contains one numpy array (saved with numpy.save function).
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

"""
with open("loadData.py") as fp:
    for i, line in enumerate(fp):
        if "\xe2" in line:
            print i, repr(line)
			"""

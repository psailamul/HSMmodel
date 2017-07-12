import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from HSM import HSM
from visualization import *


download_time = time.time() #print "Download complete: Time %s" %(time.time() - download_time)
train_input1=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_inputs.npy')
train_set1=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_set.npy')

train_input2=np.load('/home/pachaya/AntolikData/SourceCode/Data/region2/training_inputs.npy')
train_set2=np.load('/home/pachaya/AntolikData/SourceCode/Data/region2/training_set.npy')

train_input3=np.load('/home/pachaya/AntolikData/SourceCode/Data/region3/training_inputs.npy')
train_set3=np.load('/home/pachaya/AntolikData/SourceCode/Data/region3/training_set.npy')


raw_vld_set1 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/raw_validation_set.npy')
vldinput_set1 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/validation_inputs.npy')
vld_set1 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/validation_set.npy')

raw_vld_set2 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region2/raw_validation_set.npy')
vldinput_set2 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region2/validation_inputs.npy')
vld_set2 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region2/validation_set.npy')

raw_vld_set3 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region3/raw_validation_set.npy')
vldinput_set3 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region3/validation_inputs.npy')
vld_set3 = np.load('/home/pachaya/AntolikData/SourceCode/Data/region3/validation_set.npy')

#out = fitHSM(r1_train_input,r1_train_set)


#import ipdb; ipdb.set_trace()

[Ks1,hsm1] = np.load('out_region1.npy')
[Ks2,hsm2] = np.load('out_region2.npy')
[Ks3,hsm3] = np.load('out_region3.npy')



print "Download complete: Time %s" %(time.time() - download_time)

response1 = HSM.response(hsm1,train_input1,Ks1) # predicted response after train
response2 = HSM.response(hsm2,train_input2,Ks2)
response3 = HSM.response(hsm3,train_input3,Ks3)

#Validation set
pred_response1 = HSM.response(hsm1,vldinput_set1,Ks1) #predicted response for validation set
pred_response2 = HSM.response(hsm2,vldinput_set2,Ks2)
pred_response3 = HSM.response(hsm3,vldinput_set3,Ks3)

# fig 4A response vs image
# HAve to find the best neuron first
import visualization



corr1 = computeCorr(response1, train_set1)
corr2 = computeCorr(response2, train_set2)
corr3 = computeCorr(response3, train_set3)



vld_corr1 = computeCorr(pred_response1,vld_set1)
vld_corr2 = computeCorr(pred_response2,vld_set2)
vld_corr3 = computeCorr(pred_response3,vld_set3)



#(train_c, val_c) = printCorrelationAnalysis(act,val_act,pred_act,pred_val_act):

# Max corr neurons
imax1 = np.argmax(vld_corr1) # note : actually have to combine neurons in all regions
imax2 = np.argmax(vld_corr2)
imax3 = np.argmax(vld_corr3)

ax1=plt.subplot(311)
ax1.plot(vld_set1[:,imax1],'-ok')
ax1.plot(pred_response1[:,imax1],'--or')
ax1.set_title('Region 1 : Cell#%d has max corr of %f'%(imax1+1,np.max(vld_corr1)))
ax1.set_ylim([0, 6])

ax2=plt.subplot(312)
ax2.plot(vld_set2[:,imax2],'-ok')
ax2.plot(pred_response2[:,imax2],'--or')
ax2.set_title('Region 2 : Cell#%d has max corr of %f'%(imax2+1,np.max(vld_corr2)))
ax2.set_ylim([0, 6])

ax3=plt.subplot(313)
ax3.plot(vld_set3[:,imax3],'-ok')
ax3.plot(pred_response3[:,imax3],'--or')
ax3.set_title('Region 3 : Cell#%d has max corr of %f'%(imax3+1,np.max(vld_corr3)))
ax3.set_ylim([0, 6])

plt.show()

imin1 = np.argmin(vld_corr1) # note : actually have to combine neurons in all regions
imin2 = np.argmin(vld_corr2) # note : actually have to combine neurons in all regions
imin3 = np.argmin(vld_corr3) # note : actually have to combine neurons in all regions

ax1=plt.subplot(311)
ax1.plot(vld_set1[:,imin1],'-ok')
ax1.plot(pred_response1[:,imin1],'--or')
ax1.set_title('Region 1 :Cell#%d has min corr of %f'%(imin1+1,np.min(vld_corr1)))

ax2 = plt.subplot(312)
ax2.plot(vld_set2[:,imin2],'-ok')
ax2.plot(pred_response2[:,imin2],'--or')
ax2.set_title('Region 2 :Cell#%d has min corr of %f'%(imin2+1,np.min(vld_corr2)))

ax3=plt.subplot(313)
ax3.plot(vld_set3[:,imin3],'-ok')
ax3.plot(pred_response3[:,imin3],'--or')
ax3.set_title('Region 3 :Cell#%d has min corr of %f'%(imin3+1,np.min(vld_corr3)))

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

# Get predicted response from Allen Institute from the learned network from Antolik's work

#Import HSM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from HSM import HSM
from visualization import *


[Ks1,hsm1] = np.load('out_region1.npy')

# Allen Brain Institute data
train_input1=np.load('Allen_data_170513/Allen_train_input.npy')
train_set1=np.load('Allen_data_170513/Allen_train_output.npy')	

train_set1 = full_train_set1.copy()
train_set1 = train_set1[:,:103]

response1 = HSM.response(hsm1,train_input1,Ks1) # predicted response after train
vld_corr1 = computeCorr(response1, train_set1)
	
#Plot
# Max corr neurons
imax = np.argmax(vld_corr1) 

plt.plot(train_set1[:,imax],'-ok')
plt.plot(response1[:,imax],'--or')
plt.title('Cell#%d has max corr of %f'%(imax+1,np.max(vld_corr1)))
plt.show()

imin = np.argmin(vld_corr1) # note : actually have to combine neurons in all regions

plt.plot(train_set1[:,imin],'-ok')
plt.plot(response1[:,imin],'--or')
plt.title('Cell#%d has min corr of %f'%(imin+1,np.min(vld_corr1)))
plt.show()
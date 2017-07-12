from HSM import HSM
import numpy
from scipy.optimize import fmin_tnc
import param




import numpy as np
from fitting import fitHSM
import time

download_time = time.time()
#r1_train_input=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_inputs.npy')
#r1_train_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_set.npy')
training_inputs=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_inputs.npy')
training_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_set.npy')
print "Download complete: Time %s" %(time.time() - download_time)

import ipdb; ipdb.set_trace()

call_time = time.time()

seed=13; lgn=9; hlsr=0.2

num_pres,num_neurons = numpy.shape(training_set)
import ipdb; ipdb.set_trace()
print "Creating HSM model"
hsm = HSM(training_inputs,training_set) # Initialize model --> add input and output, construct parameters , build mobel, # create loss function
print "Created HSM model"   
hsm.num_lgn = lgn 
hsm.hlsr = hlsr
    
func = hsm.func() 
#import ipdb; ipdb.set_trace()
Ks = hsm.create_random_parametrization(seed) # set initial random values of the model parameter vector

#(Ks,success,c)=fmin_tnc(func ,Ks,fprime=hsm.der(),bounds=hsm.bounsd,maxfun = 100000,messages=0)  # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_tnc.html
(Ks,success,c)=fmin_tnc(func ,Ks,fprime=hsm.der(),bounds=hsm.bounds,maxfun = 100,disp=5)

print 'Final training error: ', func(numpy.array(Ks))/num_neurons/len(training_set)
runtime = time.time() - call_time
out = (Ks,hsm)
import ipdb; ipdb.set_trace()
print "Finish training %s" %(runtime)
np.save("HSMout_test.npy",out)
print "Saved"

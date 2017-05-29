import numpy as np
from fitting import fitHSM
import time

download_time = time.time()
#r1_train_input=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_inputs.npy')
#r1_train_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_set.npy')
train_input=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_inputs.npy')
train_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_set.npy')
print "Download complete: Time %s" %(time.time() - download_time)


"""
call_time = time.time()
out = fitHSM(r1_train_input,r1_train_set)
runtime = time.time() - call_time

import pdb; pdb.set_trace()
print "Finish training %s" %(runtime)
np.save("out_region_test.npy",out)
print "Saved"
"""
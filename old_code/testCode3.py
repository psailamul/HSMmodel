"""
test HSM
"""

import numpy as np
from fitting import fitHSM
import time

download_time = time.time()
r3_train_input=np.load('/home/pachaya/Desktop/Antolik Data/SourceCode/Data/region3/training_inputs.npy')
r3_train_set=np.load('/home/pachaya/Desktop/Antolik Data/SourceCode/Data/region3/training_set.npy')
print "Download complete: Time %s" %(time.time() - download_time)

call_time = time.time()
out = fitHSM(r3_train_input,r3_train_set)
runtime = time.time() - call_time
print "Finish training %s" %(runtime)
np.save("out_region3.npy",out)
print "Saved"



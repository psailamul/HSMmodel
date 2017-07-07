"""
test HSM
"""

import numpy as np
from fitting import fitHSM
import time

download_time = time.time()
r2_train_input=np.load('/home/pachaya/Desktop/Antolik Data/SourceCode/Data/region2/training_inputs.npy')
r2_train_set=np.load('/home/pachaya/Desktop/Antolik Data/SourceCode/Data/region2/training_set.npy')
print "Download complete: Time %s" %(time.time() - download_time)

call_time = time.time()
out = fitHSM(r2_train_input,r2_train_set)
runtime = time.time() - call_time
print "Finish training %s" %(runtime)
np.save("out_region2.npy",out)
print "Saved"



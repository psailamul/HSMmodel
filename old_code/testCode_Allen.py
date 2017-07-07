
import numpy as np
from fitting import fitHSM
import time


download_time = time.time()
train_input=np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_train_input.npy')
train_set=np.load('/home/pachaya/AntolikData/SourceCode/Allen_data_170513/Allen_train_output.npy')
print "Download complete: Time %s" %(time.time() - download_time)

numLGN= 5
HLSR=0.3
print "numLGN = %d, HLSR = %.1f" %(numLGN,HLSR)
call_time = time.time()
out = fitHSM(train_input,train_set,lgn=numLGN,hlsr=HLSR)
runtime = time.time() - call_time

print "Finish training %s" %(runtime)
savefname = "out_Allen_unique_lgn%d_hlsr%d.npy" %(numLGN,HLSR*10)
np.save(savefname,out)
print "Saved"

print "=================== Second Run =================="
numLGN= 15
HLSR=0.3
print "numLGN = %d, HLSR = %.1f" %(numLGN,HLSR)
call_time = time.time()
out = fitHSM(train_input,train_set,lgn=numLGN,hlsr=HLSR)
runtime = time.time() - call_time

print "Finish training %s" %(runtime)
savefname = "out_Allen_unique_lgn%d_hlsr%d.npy" %(numLGN,HLSR*10)
np.save(savefname,out)
print "Saved"

#THEANO_FLAGS=mode=FAST_RUN,device=gpu3,floatX=float32 ipython testCode_Allen.py
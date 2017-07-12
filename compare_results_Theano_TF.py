# Compare the simulation in Theano with those in TensorFlow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from HSM import HSM
from visualization import *
from get_host_path import get_host_path

# #############  Download trained result ( Antolik ) ############## 
download_time = time.time() #print "Download complete: Time %s" %(time.time() - download_time)
runpath=get_host_path()
num_region =3
train_input ={}; train_set={}; raw_vld_set={}; vldinput_set={}; vld_set={}
Ks={}; hsm={}

for i in range(num_region):
    id=str(i+1)
    train_input[id]=np.load(runpath+'Data/region'+id+'/training_inputs.npy')
    train_set[id]=np.load(runpath+'Data/region'+id+'/training_set.npy')
    raw_vld_set[id]=np.load(runpath+'Data/region'+id+'/raw_validation_set.npy')
    vldinput_set[id]=np.load(runpath+'Data/region'+id+'/validation_inputs.npy')
    vld_set[id]=np.load(runpath+'Data/region'+id+'/validation_set.npy')
    Ks[id], hsm[id] = np.load('out_region'+id+'.npy')

print "Download complete: Time %s" %(time.time() - download_time)
# #############  Get predicted response ############## 
pred_train_response={}; pred_vld_response={}

for i in range(num_region):
    id=str(i+1)
    pred_train_response[id] = HSM.response(hsm[id],train_input[id],Ks[id]) # predicted response after train
    pred_vld_response[id] = HSM.response(hsm[id],vldinput_set[id],Ks[id]) #predicted response for validation set


# ##########################################
# Plot in Figure 4
#
# #############################################

# #############  Mean activity to validation set ############## 
corr={}; vld_corr={}
for i in range(num_region):
    id=str(i+1)
    corr[id] = computeCorr(pred_train_response[id],train_set[id])
    vld_corr[id]=computeCorr(pred_vld_response[id],vld_set[id])


from funcs_for_graphs import *
compare_corr_all_regions(pred_vld_response,vld_set, vld_corr, stats_param='median', titletxt='Validation Set')

compare_corr_all_regions(pred_train_response,train_set, corr, stats_param='max', titletxt='Training Set')
compare_corr_all_regions(pred_vld_response,vld_set, vld_corr, stats_param='max', titletxt='Validation Set')




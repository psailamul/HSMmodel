#Figs for report
# 1: Reimplementation with Theano

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from HSM import HSM
from visualization import *
from get_host_path import get_host_path
import os
import re
import time
from datetime import datetime
from HSM import HSM
from funcs_for_graphs import *


# ############# Setting ###############
# ############# Setting ###############
SAVEFIG=False
dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
# ############## Specified Folder ##########################
Code='SciPy_SEEDnumpy'
#Read
DATA_LOC = '/media/data_cifs/pachaya/FYPsim/HSMmodel/TFtrainingSummary/'
data_dir =  os.path.join(DATA_LOC,'SciPy_SEEDnumpy')
#Write
FIGURES_LOC = os.path.join(os.getcwd(), 'Figures')
figures_dir = os.path.join(FIGURES_LOC,"original_model_in_theano_%s"%(dt_stamp))
try: 
    os.makedirs(figures_dir)
except OSError:
    if not os.path.isdir(figures_dir):
        raise
SAVEFIG=False
dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
# #############  Download Data Set ############## 
download_time = time.time() #print "Download complete: Time %s" %(time.time() - download_time)
runpath=os.getcwd()
NUM_REGIONS =3
train_input ={}; train_set={}; raw_vld_set={}; vldinput_set={}; vld_set={}

for i in range(NUM_REGIONS):
    id=str(i+1)
    train_input[id]=np.load(os.path.join(runpath,'Data/region'+id+'/training_inputs.npy'))
    train_set[id]=np.load(os.path.join(runpath,'Data/region'+id+'/training_set.npy'))
    raw_vld_set[id]=np.load(os.path.join(runpath,'Data/region'+id+'/raw_validation_set.npy'))
    vldinput_set[id]=np.load(os.path.join(runpath,'Data/region'+id+'/validation_inputs.npy'))
    vld_set[id]=np.load(os.path.join(runpath,'Data/region'+id+'/validation_set.npy'))

print "Download Data set complete: Time %s" %(time.time() - download_time)


rg1_train_input = train_input['1']
rg1_train_set = train_set['1']
rg1_vldinput_set = vldinput_set['1']
rg1_vld_set = vld_set['1']


##################################################
# Plot in Figure 4
##################################################

######## Download trained result #################
# seed=13, trial=0, LGN =9, hlsr=0.2
# All regions
##################################################
all_seeds = [13]
all_trials =[0]
all_regions = [1,2,3]
TN_outputs={}
TF_outputs ={}
Ks_outputs ={}
lgn=9; hlsr=0.2 #The lgn and hlsr are fixed
hsm = build_hsm_for_Theano()
data_dir_all_regions =  os.path.join(os.getcwd(),'TFtrainingSummary','SciPy_SEEDnumpy')

TN_filename = lambda seed,region:"HSMout_theano_SciPytestSeed_Rg%s_MaxIter100000_seed%g.npy"%(region,seed)
TF_filename = lambda seed,trial,region:"AntolikRegion%s_SciPy_jac_npSeed_MaxIter100000_itr1_SEED%g_trial%g_"%(region,seed,trial)
TN_list_trial = lambda seed,region:"HSMout_theano_SciPytestSeed_Rg%s_MaxIter100000_seed%g*"%(region,seed)
TF_list_trial = lambda seed,region:"AntolikRegion%s_SciPy_jac_npSeed_MaxIter100000_itr1_SEED%g_trial*"%(region,seed)
 
TN_TR_pred_response={}; TN_VLD_pred_response={};
TF_TR_pred_response={}; TF_VLD_pred_response={}

this_ss =all_seeds[0]
this_tr =all_trials[0]

for rg in all_regions:
    import ipdb; ipdb.set_trace()
    this_rg = str(rg)
    key = str(rg)
    print '[',key,']'
    #Theano
    fname = TN_filename(this_ss,this_rg) 
    print "Theano"
    print fname
    this_item = fname[:-4] #remove.npy
    tmpitem = np.load(os.path.join(data_dir,fname))
    assert tmpitem is not None
    TN_outputs[key] = tmpitem.item()
    TN_outputs[key]['hsm']=hsm[key]
    Ks_outputs[key]=TN_outputs[key]['x'] 
    #Theano
    TN_TR_pred_response[key] = HSM.response(hsm[key],train_input[key],Ks_outputs[key]) # predicted response after train
    TN_VLD_pred_response[key] = HSM.response(hsm[key],vldinput_set[key],Ks_outputs[key]) #predicted response for validation set
   
    #Tensorflow
    TF_all_folders = glob.glob(os.path.join(data_dir, TF_filename(this_ss,this_tr,this_rg) +'*'))
    print TF_all_folders
    for this_item in TF_all_folders: 
        tmpdat=load_TensorFlow_outputs('', data_dir, this_item,split_path=False)
        if tmpdat is not None:
            break
    if tmpdat is None:
        print "Error: File not found\t Region = %s"%(key)
        print this_item
    TF_outputs[key]  = tmpdat
    #TensorFlow
    TF_TR_pred_response[key] = TF_outputs[key]['TR_1st_pred_response'] # predicted response after train
    TF_VLD_pred_response[key] = TF_outputs[key]['VLD_1st_ypredict'] #predicted response for validation set
# #############  Mean activity to validation set ############## 
import ipdb; ipdb.set_trace()

TN_corr={}; TN_vld_corr={}
TF_corr={}; TF_vld_corr={}

for i in range(NUM_REGIONS):
    id=str(i+1)
    TN_corr[id] = computeCorr(TN_TR_pred_response[id],train_set[id])
    TN_vld_corr[id]=computeCorr(TN_VLD_pred_response[id],vld_set[id])
    TF_corr[id] = computeCorr(TF_TR_pred_response[id],train_set[id])
    TF_vld_corr[id]=computeCorr(TF_VLD_pred_response[id],vld_set[id])

#combine response from all region
for rg in all_regions:
    #Theano
    fname = TN_filename(this_ss,this_rg)
    this_item = fname[:-4] #remove.npy
    check_file = glob.glob(os.path.join(data_dir,fname))
    if check_file is None:
        print "Error: File not found\t Theano seed = %g"%(ss)
        print this_item
    TF_all_folders = glob.glob(os.path.join(data_dir, TF_filename(this_ss,this_rg) +'*'))
    for this_item in TF_all_folders: 
        print "\t\t this item: ",this_item
        tmpdat=load_TensorFlow_outputs('', data_dir, this_item,split_path=False)
        if tmpdat is not None:
            break
    if tmpdat is None:
        print "Error: File not found\t Tensorflow seed = %g"%(ss)
        print TF_all_folders


# Compare the simulation in Theano with those in TensorFlow
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


# # ############# Functions ###############
def get_param_from_fname(fname, keyword):
    cuts = re.split('_',fname)
    for prm in cuts:
        if str.startswith(prm,keyword):
            return prm[len(keyword):]
    else:
        print "WARNING:: KEYWORD NOT FOUND"
        return None
def load_TensorFlow_outputs(current_path, data_dir, dir_item):
    dir_item+='/'
    directory = current_path+data_dir+dir_item
    TF_DAT={}
    for root, dirs, files in os.walk(directory): 
      # Look inside folder
      matching = [fl for fl in files if fl.endswith('.npz') ]
      if len(matching) == 0:
        continue
      fname=matching[0]
      fullpath = data_dir+dir_item+fname
      npz_dat = np.load(fullpath)
      for k in npz_dat.keys():
        if k == 'CONFIG':
            TF_DAT[k]=npz_dat[k].item()
        else:
            TF_DAT[k]=npz_dat[k]
      npz_dat.close()
      return TF_DAT
    return None

def build_hsm_for_Theano(REGION_NUM=3, seed=13, lgn=9, hlsr=0.2): # May not need
    rg =1
    from HSM import HSM
    import param
    import numpy as np
    from get_host_path import get_host_path
    ALL_HSM={}
    PATH=get_host_path(HOST=False, PATH=True)
    while rg <= REGION_NUM:
        Region_num=str(rg)
        training_inputs=np.load(PATH+'Data/region'+Region_num+'/training_inputs.npy')
        training_set=np.load(PATH+'Data/region'+Region_num+'/training_set.npy')
        num_pres,num_neurons = np.shape(training_set)
        print "Creating HSM model"
        hsm = HSM(training_inputs,training_set) # Initialize model --> add input and output, construct parameters , build mobel, # create loss function
        print "Created HSM model"   
        hsm.num_lgn = lgn 
        hsm.hlsr = hlsr
        ALL_HSM[Region_num]=hsm
        rg+=1
    return ALL_HSM
    
def combine_corrs(corrs):
    return np.concatenate((corrs['1'], corrs['2'],corrs['3']),axis=0)

def combine_responses(response):
    return np.concatenate((response['1'], response['2'],response['3']),axis=1) 
    
# ############# Setting ###############
SAVEFIG=True
SEED=13
FIG_HANDLES=[]
FIG_NAMES=[]

# ############## Specified Folder ##########################
Code='TN_TF_comparison_new'
PATH = '/media/data_cifs/pachaya/SourceCode'
#TFtrainingSummary/SciPy_SEEDnumpy/AntolikRegion3_SciPy_jac_npSeed_MaxIter100000_itr2_SEED13_2017_07_16_00_02_57/TRdat_trainedHSM_region3_trial13.npz

HOST='x8'

SUMMARY_DIR = 'TFtrainingSummary/SciPy_SEEDnumpy/'

current_path = PATH
data_dir = os.path.join(  current_path, "TFtrainingSummary/SciPy_SEEDnumpy/")

all_dirs = os.listdir(data_dir)

# Save Figure
if SAVEFIG :
    date=str(datetime.now())
    date = date[:10]
    if not os.path.isdir(current_path+'/Figures/'+date+'_'+Code+'/') :
      os.mkdir(current_path+'/Figures/'+date+'_'+Code+'/')
    Fig_fold=current_path+'/Figures/'+date+'_'+Code+'/'
    dt_stamp = re.split(
      '\.', str(datetime.now()))[0].\
      replace(' ', '_').replace(':', '_').replace('-', '_')

# #############  Download Data Set ############## 
download_time = time.time() #print "Download complete: Time %s" %(time.time() - download_time)
runpath=get_host_path()
NUM_REGIONS =3
train_input ={}; train_set={}; raw_vld_set={}; vldinput_set={}; vld_set={}

for i in range(NUM_REGIONS):
    id=str(i+1)
    train_input[id]=np.load(runpath+'Data/region'+id+'/training_inputs.npy')
    train_set[id]=np.load(runpath+'Data/region'+id+'/training_set.npy')
    raw_vld_set[id]=np.load(runpath+'Data/region'+id+'/raw_validation_set.npy')
    vldinput_set[id]=np.load(runpath+'Data/region'+id+'/validation_inputs.npy')
    vld_set[id]=np.load(runpath+'Data/region'+id+'/validation_set.npy')

print "Download Data set complete: Time %s" %(time.time() - download_time)


# ####### Download trained result ################
Theano_outputs={}; TensorFlow_outputs={}
for i in range(NUM_REGIONS):
    Theano_outputs[str(i+1)]=None
    TensorFlow_outputs[str(i+1)]=None

hsm = build_hsm_for_Theano()
Ks={}
for dir_item in all_dirs:
    if str.startswith(dir_item,'HSMout_theano_SciPytestSeed'): #  Theano # HSMout_theano_SciPytestSeed_Rg1_MaxIter100000_seed13
      rg_id=get_param_from_fname(dir_item, 'Rg'); 
      tmpitem = np.load(data_dir+dir_item)
      Theano_outputs[rg_id]=tmpitem.item()
      Theano_outputs[rg_id]['hsm']=hsm[rg_id]
      Ks[rg_id]=Theano_outputs[rg_id]['x']
    elif str.startswith(dir_item,'AntolikRegion'): # Tensorflow
      rg_id=get_param_from_fname(dir_item, 'AntolikRegion'); 
      tmpdat=load_TensorFlow_outputs('', data_dir, dir_item)
      assert tmpdat is not None 
      TensorFlow_outputs[rg_id]=tmpdat
    else:
      continue

# #############  Get predicted response ############## 
Theano_TR_pred_response={}; Theano_VLD_pred_response={};
TF_TR_pred_response={}; TF_VLD_pred_response={}

for i in range(NUM_REGIONS):
    id=str(i+1)
    #Theano
    Theano_TR_pred_response[id] = HSM.response(hsm[id],train_input[id],Ks[id]) # predicted response after train
    Theano_VLD_pred_response[id] = HSM.response(hsm[id],vldinput_set[id],Ks[id]) #predicted response for validation set
    #TensorFlow
    TF_TR_pred_response[id] = TensorFlow_outputs[id]['TR_1st_pred_response'] # predicted response after train
    TF_VLD_pred_response[id] = TensorFlow_outputs[id]['VLD_1st_ypredict'] #predicted response for validation set

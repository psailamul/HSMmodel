#analyse revise HSM

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

    
#main

sim_result_dir = '/media/data_cifs/pachaya/FYPsim/FYP/TFtrainingSummary'
EXP_LIST=['SciPy_jac_npSeed', 'Adam', 'Antolik_CONV9', 'Allen_HSM_1e-4', 'Allen_CONV5', 'Allen_CONV9', 'Allen_HSM'];

# ############## Specified Folder ##########################

DATA_LOC = '/media/data_cifs/pachaya/FYPsim/HSMmodel/TFtrainingSummary/'
exp_code ='Adam'
data_dir =  os.path.join(DATA_LOC,exp_code) # EXP LIST
FIGURES_LOC = os.path.join(os.getcwd(), 'Figures')
figures_dir = os.path.join(FIGURES_LOC,"compare_seeds_%s"%(dt_stamp))

TF_filename = lambda code, seed,trial:"AntolikRegion1_%s_MaxIter100000_itr1_SEED%g_trial%g_"%(code,seed,trial)
TF_list_trial = lambda code,seed:"AntolikRegion1_%s_MaxIter100000_itr1_SEED%g_trial*"%(code,seed)
seed_list = np.arange(30)+1

all_dirs=os.lisrdir(data_dir)
for ss in seed_list:
    folname = TF_list_trial(exp_code,ss)
    all_folders = glob.glob(os.path.join(data_dir, folname))
    for this_item in all_folders: #in case there are multiple trials/run
        tmpdat=load_TensorFlow_outputs('', data_dir, this_item,split_path=False)
        if tmpdat is not None:
            break
    if tmpdat is None:
        print "Error: File not found\t Tensorflow seed = %g"%(ss)
        print this_item
    TF_across_seeds[str(ss)]  = tmpdat


########################################################################
# Check same seed with different trials 
########################################################################
    
for ss in seed_list:
    #Theano
    fname = TN_filename(ss,curr_trial) 
    this_item = fname[:-4] #remove.npy
    tmpitem = np.load(os.path.join(data_dir,fname))
    TN_across_seeds[str(ss)] = tmpitem.item()
    TN_across_seeds[str(ss)]['hsm']=hsm
    Ks_across_seeds[str(ss)]=TN_across_seeds[str(ss)]['x']
    #Tensorflow
    TF_all_folders = glob.glob(os.path.join(data_dir, TF_filename(ss,curr_trial) +'*'))
    for this_item in TF_all_folders: 
        tmpdat=load_TensorFlow_outputs('', data_dir, this_item,split_path=False)
        if tmpdat is not None:
            break
    if tmpdat is None:
        print "Error: File not found\t Tensorflow seed = %g"%(ss)
        print this_item
    TF_across_seeds[str(ss)]  = tmpdat

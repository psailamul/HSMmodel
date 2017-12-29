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

    
    
   
# ############# Setting ###############
SAVEFIG=False
dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
Code='SciPy_SEEDnumpy'

# ############## Specified Folder ##########################

DATA_LOC = '/media/data_cifs/pachaya/FYPsim/HSMmodel/TFtrainingSummary/'
data_dir =  os.path.join(DATA_LOC,'SciPy_SEEDnumpy')
FIGURES_LOC = os.path.join(os.getcwd(), 'Figures')
figures_dir = os.path.join(FIGURES_LOC,"compare_seeds_%s"%(dt_stamp))

SEED_LIST = np.arange(50) +1
TRIAL_LIST = 0







all_dirs = os.listdir(current_path+data_dir)


import glob, os
os.chdir("/mydir")
for file in glob.glob("*.txt"):
    print(file)
or simply os.listdir:

import os
for file in os.listdir("/mydir"):
    if file.endswith(".txt"):
        print(os.path.join("/mydir", file))
or if you want to traverse directory, use os.walk:

import os

for root, dirs, files in os.walk(data_dir):
    import ipdb; ipdb.set_trace();
    for file in files:
        if file.endswith(".npy"):
             print(os.path.join(root, file))
             
             
    
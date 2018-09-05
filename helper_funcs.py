import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import re
import time
from datetime import datetime

def get_param_from_fname(fname, keyword, REPORT=True):
    cuts = re.split('_',fname)
    for prm in cuts:
        if str.startswith(prm,keyword):
            return prm[len(keyword):]
    else:
        if REPORT:
            print "WARNING:: KEYWORD NOT FOUND"
        return None
        
def load_TensorFlow_outputs(current_path='', data_dir='', dir_item='',split_path = True, import_data=True):
    if not str.endswith(dir_item,'/'):
        dir_item = "%s/"%(dir_item)
    if not split_path:
        directory = dir_item
    else:
        directory = os.path.join(current_path,data_dir,dir_item)
    
    TF_DAT={}
    for root, dirs, files in os.walk(directory): 
        # Look inside folder
        matching = [fl for fl in files if fl.endswith('.npz') ]
        if len(matching) == 0:
            continue
        fname=matching[0]
        if import_data:
            fullpath = directory+fname
            npz_dat = np.load(fullpath)
            for k in npz_dat.keys():
                if k == 'CONFIG':
                    TF_DAT[k]=npz_dat[k].item()
                else:
                    TF_DAT[k]=npz_dat[k]
            npz_dat.close()
            return TF_DAT
        else:
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

def plot_training_stats(trained_data, titletxt='', SHOW=True,RETURN=False,fontsize=14,figsize=[12,8]):
    trdat = trained_data
    iterations = trdat['STOP_idx']
    loss_list = trdat['TR_loss']
    MSE_list = trdat['TR_MSE']
    corr_list = trdat['TR_corr']
    yhat_std = trdat['TR_std_pred_response']

    if titletxt =='':
        titletxt="%s"%(trdat['CONFIG']['runcodestr'])
    fig, ax = plt.subplots(figsize=figsize)
    
    itr_idx = range(iterations)
    plt.subplot(2, 2, 1)
    plt.plot(itr_idx, loss_list, '-ok')
    plt.title('Loss', size=fontsize) 
    plt.xlabel('iterations')

    plt.subplot(2, 2, 2)
    plt.plot(itr_idx, MSE_list, '-ob')
    plt.title('MSE', size=fontsize)
    plt.xlabel('iterations')

    plt.subplot(2, 2, 3)
    plt.plot(itr_idx, corr_list, '-or')
    plt.title('Mean Correlation', size=fontsize)
    plt.xlabel('iterations')

    plt.subplot(2, 2, 4)
    plt.plot(itr_idx, yhat_std, '-ok')
    plt.title('std of predicted response', size=fontsize)
    plt.xlabel('iterations')
        
    plt.suptitle("%s"%(titletxt))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if RETURN:
        return fig #Note: don't show here
    if SHOW:
        plt.show()
        
def plot_param_table(numLGN_candidates,hlsr_candidates,this_table,title='Parameter space',SHOW=False,cm='hot'):
    plt.figure() 
    if hlsr_candidates[0] !=0:
        tmp = np.zeros(len(hlsr_candidates)+1)
        tmp[1:] = hlsr_candidates
        hlsr_candidates=tmp
    if numLGN_candidates[0] !=0:
        tmp = np.zeros(len(numLGN_candidates)+1)
        tmp[1:] = numLGN_candidates
        numLGN_candidates=tmp    
    X, Y = np.meshgrid(numLGN_candidates,hlsr_candidates)
    nonz = this_table[np.where(this_table!=0)].flatten()
    minv = np.min(nonz); maxv=np.max(nonz)
    plt.pcolor(X, Y,this_table, cmap=cm, vmin=minv,vmax=maxv)
    plt.colorbar(orientation='horizontal')
    plt.yticks(label=hlsr_candidates, fontsize=9)
    plt.ylabel('Hidden layer threshold (hlsr)')
    plt.xticks(label=numLGN_candidates, fontsize=9)
    plt.xlabel('# LGN')
    plt.title(title)
    if SHOW:
        plt.show()   

def check_all_seed(DATA_LOC, exp_code, exp_fold, region=1):
    print "============================"
    print exp_code
    print "============================"
    exp_results=[]
    data_dir =  os.path.join(DATA_LOC,exp_code)
    all_dirs = os.listdir(data_dir)
    if not all_dirs:
        print "No training result"
        return None
    for dir_item in all_dirs:
        tr=''
        if dir_item.endswith('.npz'):
            ss=get_param_from_fname(dir_item, 'region'+str(region) +'seed')
            tr=get_param_from_fname(dir_item, 'trial',REPORT=False) 
            print "Found data for seed=%s"%(ss)
            exp_results.append(ss)    
        elif str.startswith(dir_item,"%s"%(exp_fold)): # Tensorflow
            ss=get_param_from_fname(dir_item, 'SEED',REPORT=False) 
            if ss is None:
                ss=get_param_from_fname(dir_item, 'seed', REPORT=False)
            tr=get_param_from_fname(dir_item, 'trial',REPORT=False) 
            tmpdat=load_TensorFlow_outputs('','', os.path.join(data_dir, dir_item),split_path=False, import_data=False)
            if tmpdat is None:
                print "data file not found"
                print  dir_item
            else:
                if ss in exp_results:
                    k = "%s-%s"%(ss,tr)
                    if k not in exp_results:
                        exp_results.append(k)
                        print "k: ",k
                else:
                    exp_results.append(ss)
                print "Found data for seed=%s trial=%s"%(ss,tr)
                
    return exp_results
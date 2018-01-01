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
def concat_flatten_regions(input_data):
    return_data = []
    for k,v in input_data.iteritems():
        if len(v.shape) >1:
            input_data[k] = v.flatten()
        return_data.append(input_data[k])
        print k
        print v.shape
        print "---------"
        
    return np.concatenate(return_data)
    
def concat_regions(input_data):
    return_data = np.concatenate((input_data['1'],input_data['2']), axis=1)
    return_data = np.concatenate((return_data,input_data['3']), axis=1)
    
    return return_data
# ############# Setting ###############
SAVEFIG=False
SEED=13
# ############## Specified Folder ##########################
Code='SciPytestSeed'
HOST, PATH = get_host_path(HOST=True, PATH=True)
SUMMARY_DIR = 'TFtrainingSummary/SciPy_SEEDnumpy/'

current_path = os.getcwd()+'/'
data_dir = os.path.join(  "TFtrainingSummary/SciPy_SEEDnumpy/")

all_dirs = os.listdir(current_path+data_dir)

# Save Figure
if SAVEFIG :
    date=str(datetime.now())
    date = date[:10]
    if not os.path.isdir('Figures/'+date+'_'+Code+'/') :
      os.mkdir('Figures/'+date+'_'+Code+'/')
    Fig_fold='Figures/'+date+'_'+Code+'/'
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
      tmpitem = np.load(current_path+data_dir+dir_item)
      Theano_outputs[rg_id]=tmpitem.item()
      Theano_outputs[rg_id]['hsm']=hsm[rg_id]
      Ks[rg_id]=Theano_outputs[rg_id]['x']
    elif str.startswith(dir_item,'AntolikRegion'): # Tensorflow
      rg_id=get_param_from_fname(dir_item, 'AntolikRegion'); 
      tmpdat=load_TensorFlow_outputs(current_path, data_dir, dir_item)
      if tmpdat is not None:
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

# #################################################
# Plot in Figure 4
# #################################################

# #############  Mean activity to validation set ############## 
TN_corr={}; TN_vld_corr={}
TF_corr={}; TF_vld_corr={}

for i in range(NUM_REGIONS):
    id=str(i+1)
    TN_corr[id] = computeCorr(Theano_TR_pred_response[id],train_set[id])
    TN_vld_corr[id]=computeCorr(Theano_VLD_pred_response[id],vld_set[id])
    TF_corr[id] = computeCorr(TF_TR_pred_response[id],train_set[id])
    TF_vld_corr[id]=computeCorr(TF_VLD_pred_response[id],vld_set[id])
    
# train_set_all = concat_flatten_regions(train_set)
# TN_TR_pred_response_all = concat_flatten_regions(Theano_TR_pred_response)
# TF_TR_pred_response_all = concat_regions(TF_TR_pred_response)
# TN_all_corr =concat_regions(TN_corr)
# TF_all_corr =concat_regions(TF_corr)

vld_set_all = concat_regions(vld_set)
TN_VLD_pred_response_all = concat_regions(Theano_VLD_pred_response)
TF_VLD_pred_response_all = concat_regions(TF_VLD_pred_response)
TN_vld_all_corr =concat_flatten_regions(TN_vld_corr)
TF_vld_all_corr =concat_flatten_regions(TF_vld_corr)



plot_act_of_max_min_corr(runcodestr='Original HSM', 
                        yhat=TN_VLD_pred_response_all,
                        train_set=vld_set_all,
                        corr=TN_vld_all_corr, 
                        PLOT=True,
                        ZOOM=False)
                        
plot_act_of_max_min_corr(runcodestr='Reimplemented version', 
                        yhat=TF_VLD_pred_response_all,
                        train_set=vld_set_all,
                        corr=TF_vld_all_corr, 
                        PLOT=True,
                        ZOOM=False)


plot_corr_response_scatter(pred_response=TN_VLD_pred_response_all, 
                vld_set=vld_set_all, 
                corr_set=TN_vld_all_corr, 
                stats_param='median',
                titletxt='Original HSM', 
                RETURN=False, 
                datalabel1='Measured Response', 
                datalabel2='Predicted Response')
                
plot_corr_response_scatter(pred_response=TF_VLD_pred_response_all, 
                vld_set=vld_set_all, 
                corr_set=TF_vld_all_corr, 
                stats_param='median',
                titletxt='Reimplemented version', 
                RETURN=False, 
                datalabel1='Measured Response', 
                datalabel2='Predicted Response')
    


#Theano : Training Set
Tn_TR_fig_max=compare_corr_all_regions(Theano_TR_pred_response,train_set, TN_corr, stats_param='max', titletxt='Theano :: Training Set', RETURN=True)
Tn_TR_fig_med=compare_corr_all_regions(Theano_TR_pred_response,train_set, TN_corr, stats_param='median', titletxt='Theano :: Training Set', RETURN=True)

#Theano : Validation Set
Tn_VLD_fig_max=compare_corr_all_regions(Theano_VLD_pred_response,vld_set, TN_vld_corr, stats_param='max', titletxt='Theano :: Validation Set', RETURN=True)
Tn_VLD_fig_med=compare_corr_all_regions(Theano_VLD_pred_response,vld_set, TN_vld_corr, stats_param='median', titletxt='Theano :: Validation Set', RETURN=True)

#TensorFlow : Training Set
TF_TR_fig_max=compare_corr_all_regions(TF_TR_pred_response,train_set, TF_corr, stats_param='max', titletxt='TF:: 1st Training Set', RETURN=True)
TF_TR_fig_med=compare_corr_all_regions(TF_TR_pred_response,train_set, TF_corr, stats_param='median', titletxt='TF:: 1st Training Set', RETURN=True)

#TensorFlow : Validation Set
TF_VLD_fig_max=compare_corr_all_regions(TF_VLD_pred_response,vld_set, TF_vld_corr, stats_param='max', titletxt='TF :: 1st Validation Set', RETURN=True)
TF_VLD_fig_med=compare_corr_all_regions(TF_VLD_pred_response,vld_set, TF_vld_corr, stats_param='median', titletxt='TF:: 1st Validation Set', RETURN=True)

"""
#TN & TF corr 
TNTF_corr={}; TNTF_vld_corr={}
for i in range(NUM_REGIONS):
    id=str(i+1)
    TNTF_corr[id] = computeCorr(Theano_TR_pred_response[id],TF_TR_pred_response[id])
    TNTF_vld_corr[id]=computeCorr(Theano_VLD_pred_response[id],TF_VLD_pred_response[id])

TNTF_TR_fig_max=compare_corr_all_regions(Theano_TR_pred_response,TF_TR_pred_response, TNTF_corr, stats_param='max', titletxt='Theano&TensorFlow :: Training Set', RETURN=True)
TNTF_VLD_fig_max=compare_corr_all_regions(Theano_VLD_pred_response,TF_VLD_pred_response, TNTF_vld_corr, stats_param='max', titletxt='Theano&TensorFlow :: Validation Set', RETURN=True)

TNTF_TR_fig_max=compare_corr_all_regions(Theano_TR_pred_response,TF_TR_pred_response, TNTF_corr, stats_param='median', titletxt='Theano&TensorFlow :: Training Set', RETURN=True)
TNTF_VLD_fig_max=compare_corr_all_regions(Theano_VLD_pred_response,TF_VLD_pred_response, TNTF_vld_corr, stats_param='median', titletxt='Theano&TensorFlow :: Validation Set', RETURN=True)


#Histogram of TN & TF correlation
combine_TNTF_corr = np.concatenate((TNTF_corr['1'], TNTF_corr['2'],TNTF_corr['3']),axis=0)
combine_TNTF_vld_corr = np.concatenate((TNTF_vld_corr['1'], TNTF_vld_corr['2'],TNTF_vld_corr['3']),axis=0)

plt.subplot(121); plt.hist(combine_TNTF_corr,normed=True); plt.xlim([0,1]); plt.title('Training set')
plt.subplot(122); plt.hist(combine_TNTF_vld_corr,normed=True); plt.xlim([0,1]); plt.title('Validation set')
plt.suptitle('Distribution of correlation coefficient between response from Theano and TensorFlow')
"""

# #############Combine all regions :: cdf ############## 
# cumsum 

fig_TNTR,Xs,Fs = cdf_allregions( TN_corr, NUM_REGIONS=3, DType='Theano Training set : ', C_CODE=True, SHOW=True, RETURN=True)
fig_TNVLD,Xs,Fs = cdf_allregions( TN_vld_corr, NUM_REGIONS=3, DType='Theano Validation set : ', C_CODE=True, SHOW=True, RETURN=True)

fig_TFTR,Xs,Fs = cdf_allregions( TF_corr, NUM_REGIONS=3, DType='TensorFlow Training set : ', C_CODE=True, SHOW=True, RETURN=True)
fig_TFVLD,Xs,Fs = cdf_allregions( TF_vld_corr, NUM_REGIONS=3, DType='TensorFlow Validation set : ', C_CODE=True, SHOW=True, RETURN=True)

"""
fig_TNTF_TR,Xs,Fs = cdf_allregions( TNTF_corr, NUM_REGIONS=3, DType='Training set of Theano and TensorFlow version: ', C_CODE=True, SHOW=True, RETURN=True)
fig_TNTF_VLD,Xs,Fs = cdf_allregions( TNTF_vld_corr, NUM_REGIONS=3, DType='Validation set of Theano and TensorFlow version: ', C_CODE=True, SHOW=True, RETURN=True)

 
if False:
    #For FYP 
    #Data , Theano, TF 
    # Cell with median corr between TN and TF Median Cell#130 idx=139
    #Label TN with data 
    # Label TF with data
    Theano_VLD_pred_response,vld_set, TN_vld_corr
    TF_VLD_pred_response,vld_set, TF_vld_corr


    # vld_set, 
    # Antolik_set
    # TF_set = 
    # N1=len(corr_set);
    # idx1 = N1/2-1 if N1%2==0 else (N1-1)/2
    # stat1=np.sort(corr_set)[idx1]
    # datalabel1='Measured Response'
    # datalabel2='Predicted Response'

    fig, ax = plt.subplots()
    plt.subplot(2,1,1)
    plt.plot(vld_set[:,idx1],'-ok',label='Measured Response')
    plt.plot(pred_response[:, idx1],'--or',label='Predicted Response')
    plt.plot(pred_response[:, idx1],'--or',label='Predicted Response')

    plt.ylabel('Response')
    plt.xlabel('Image #')
    plt.title("Cell#%g is the %s  neuron, R = %.5f, mean neuron has R = %.5f"%(idx1+1 ,stats_param,stat1, np.mean(corr_set)))
    plt.legend(loc=0)

    plt.subplot(2,1,2)
    plt.scatter(vld_set[:,idx1], pred_response[:,idx1])
    N=np.ceil(np.max([np.max(pred_response[:,idx1]),np.max(vld_set[:,idx1])]))
    plt.plot(np.arange(N),np.arange(N),'--c')
    plt.xlim([0,N]); plt.ylim([0,N])
    plt.ylabel(datalabel2)
    plt.xlabel(datalabel1)
    plt.title('Scatter plot of measured response and predicted response of cell#%g'%(idx1+1))


    if(SAVEFIG):
        Tn_TR_fig_max.savefig(Fig_fold+"Tn_SEED%g_TR_max.png"%(SEED))
        Tn_TR_fig_med.savefig(Fig_fold+"Tn_SEED%g_TR_med.png"%(SEED))
        Tn_VLD_fig_max.savefig(Fig_fold+"Tn_SEED%g_VLD_max.png"%(SEED))
        Tn_VLD_fig_med.savefig(Fig_fold+"Tn_SEED%g_VLD_med.png"%(SEED))

        TF_TR_fig_max.savefig(Fig_fold+"TF_SEED%g_TR_max.png"%(SEED))
        TF_TR_fig_med.savefig(Fig_fold+"TF_SEED%g_TR_med.png"%(SEED))
        TF_VLD_fig_max.savefig(Fig_fold+"TF_SEED%g_VLD_max.png"%(SEED))
        TF_VLD_fig_med.savefig(Fig_fold+"TF_SEED%g_VLD_med.png"%(SEED))



#    fig_hist.savefig(Fig_fold+sim_folder+sim_code+"_ResponseDist.png")
"""
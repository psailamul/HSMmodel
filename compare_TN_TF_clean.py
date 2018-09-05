"""
All analysis

"""
import funcs_for_graphs as hp #Helper functions
from datetime import datetime
from scipy import stats
from HSM import HSM
import numpy as np
import param
import glob
import os
import re

########################################################################
# Setting
########################################################################
SAVEFIG=False
dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
hsm = build_hsm_for_Theano()

# #############  Download The Antolik Data Set ############## 
download_time = time.time() #print "Download complete: Time %s" %(time.time() - download_time)
runpath=os.getcwd()
id ='1'
rg1_train_input = np.load(os.path.join(runpath,'Data/region'+id+'/training_inputs.npy'))
rg1_train_set = np.load(os.path.join(runpath,'Data/region'+id+'/training_set.npy'))
rg1_vldinput_set = np.load(os.path.join(runpath,'Data/region'+id+'/validation_inputs.npy'))
rg1_vld_set = np.load(os.path.join(runpath,'Data/region'+id+'/validation_set.npy'))
print "Download Data set complete: Time %s" %(time.time() - download_time)

# ############## Specified Folder ##########################
DATA_LOC = '/media/data_cifs/pachaya/FYPsim/HSMmodel/TFtrainingSummary/'
data_dir =  os.path.join(DATA_LOC,'SciPy_SEEDnumpy')
FIGURES_LOC = os.path.join(os.getcwd(), 'Figures')
figures_dir = os.path.join(FIGURES_LOC,"compare_seeds_%s"%(dt_stamp))

TN_filename = lambda seed,trial:"HSMout_theano_SciPytestSeed_Rg1_MaxIter100000_seed%g-%g.npy"%(seed,trial)
TF_filename = lambda seed,trial:"AntolikRegion1_SciPy_jac_npSeed_MaxIter100000_itr1_SEED%g_trial%g_"%(seed,trial)


########################################################################
# Download training result from both TN and TF 
########################################################################

seed_list = np.arange(50) +1
curr_trial = 0
TN_across_seeds ={}
Ks_across_seeds ={}
TF_across_seeds ={}
lgn=9; hlsr=0.2
hsm = HSM(rg1_train_input,rg1_train_set) 
hsm.num_lgn = lgn 
hsm.hlsr = hlsr      
for ss in seed_list:
    print "ss:",ss
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
#Get responses
Theano_TR_pred_response= {}
Theano_VLD_pred_response={}
TF_TR_pred_response={}
TF_VLD_pred_response={}

hsm_tr = HSM(rg1_train_input,rg1_train_set)  
hsm_tr.num_lgn = lgn 
hsm_tr.hlsr = hlsr
hsm_vld = HSM(rg1_vldinput_set,rg1_vld_set) 
hsm_vld.num_lgn = lgn 
hsm_vld.hlsr = hlsr

for ss in seed_list:
    #Theano
    Theano_TR_pred_response[str(ss)] = HSM.response(hsm_tr,rg1_train_input,Ks_across_seeds[str(ss)]) # predicted response after train
    Theano_VLD_pred_response[str(ss)] = HSM.response(hsm_vld,rg1_vldinput_set,Ks_across_seeds[str(ss)]) #predicted response for validation set
    #TensorFlow
    TF_TR_pred_response[str(ss)] = TF_across_seeds[str(ss)]['TR_1st_pred_response'] # predicted response after train
    TF_VLD_pred_response[str(ss)] = TF_across_seeds[str(ss)]['VLD_1st_ypredict'] #predicted response for validation set

# #############  Mean activity to validation set ############## 
TN_corr=np.zeros([50,103]); 
TN_vld_corr=np.zeros([50,103])
TF_corr=np.zeros([50,103]); 
TF_vld_corr=np.zeros([50,103])
ttest_result_TR=[]
pval_TR=np.zeros(50)
ttest_result_VLD=[]
pval_VLD=np.zeros(50)
pval_direct_TR=np.zeros(50)
pval_direct_VLD=np.zeros(50)
for ss in seed_list:
    id=str(ss)
    print "id: ", id
    TN_corr[ss-1,:] = computeCorr(Theano_TR_pred_response[id],rg1_train_set) #corr per neuron
    TN_vld_corr[ss-1,:]=computeCorr(Theano_VLD_pred_response[id],rg1_vld_set)
    TF_corr[ss-1,:] = computeCorr(TF_TR_pred_response[id],rg1_train_set)
    TF_vld_corr[ss-1,:]=computeCorr(TF_VLD_pred_response[id],rg1_vld_set)   
    ttestTR= ttest_in(TN_corr[id],TF_corr[id])
    ttest_result_TR.append(ttestTR)
    pval_TR[ss-1] = ttestTR.pvalue
    ttestVLD= ttest_ind(TN_vld_corr[id],TF_vld_corr[id])
    ttest_result_VLD.append(ttestVLD)
    pval_VLD[ss-1]= ttestVLD.pvalue
    tmpTR= ttest_rel(Theano_TR_pred_response[id].flatten(),TF_TR_pred_response[id].flatten())
    pval_direct_TR[ss-1]=tmpTR.pvalue
    tmpVLD= ttest_rel(Theano_VLD_pred_response[id].flatten(),TF_VLD_pred_response[id].flatten())
    pval_direct_VLD[ss-1]=tmpVLD.pvalue
resTR=ttest_rel(TN_corr,TF_corr)
resVLD=ttest_rel(TN_vld_corr,TF_vld_corr)
#Remove Cell ID 67
best_seed=np.argmax(np.mean(TN_corr,axis=1))
#============================================================================
R2_training=[]
R2_VLD =[]
onebig_TN_TR=np.zeros([1800,103,50]) #3D: Image x neuron x seed
onebig_TF_TR=np.zeros([1800,103,50])
onebig_TN_VLD=np.zeros([50,103,50])
onebig_TF_VLD=np.zeros([50,103,50])

for ss in seed_list:
    id = str(ss)
    print id
    #Training set
    setname ='TRAINING'
    response1 =Theano_TR_pred_response[id]
    response2 =TF_TR_pred_response[id]
    r_sqr = TN_TF_Rsquare(response1, response2)
    R2_training.append(r_sqr)
    #Test set
    setname ='TEST'
    response1 =Theano_VLD_pred_response[id]
    response2 =TF_VLD_pred_response[id]
    r_sqr = TN_TF_Rsquare(response1, response2)
    R2_VLD.append(r_sqr)
    onebig_TN_TR[:,:,ss-1]= Theano_TR_pred_response[id]
    onebig_TF_TR[:,:,ss-1]=TF_TR_pred_response[id]
    onebig_TN_VLD[:,:,ss-1]=Theano_VLD_pred_response[id]
    onebig_TF_VLD[:,:,ss-1]=TF_VLD_pred_response[id]

allseeds_TN_TR=np.zeros([50,1800*103]) #2D: seed x activity of a neuron per image(=imagexneuron)
allseeds_TF_TR=np.zeros([50,1800*103])
allseeds_TN_VLD=np.zeros([50,50*103])
allseeds_TF_VLD=np.zeros([50,50*103])
ttest_result_TR=[]
pval_TR=np.zeros(50)
ttest_result_VLD=[]
pval_VLD=np.zeros(50)
for ss in seed_list:
    id = str(ss)
    allseeds_TN_TR[ss-1,:]= Theano_TR_pred_response[id].flatten()
    allseeds_TF_TR[ss-1,:]=TF_TR_pred_response[id].flatten()
    allseeds_TN_VLD[ss-1,:]=Theano_VLD_pred_response[id].flatten()
    allseeds_TF_VLD[ss-1,:]=TF_VLD_pred_response[id].flatten()
    ttestTR= ttest_rel(Theano_TR_pred_response[id].flatten(),TF_TR_pred_response[id].flatten())
    ttest_result_TR.append(ttestTR)
    pval_TR[ss-1] = ttestTR.pvalue
    ttestVLD= ttest_rel(Theano_VLD_pred_response[id].flatten(),TF_VLD_pred_response[id].flatten())
    ttest_result_VLD.append(ttestVLD)
    pval_VLD[ss-1]= ttestVLD.pvalue
        
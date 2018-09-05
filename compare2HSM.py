##############################################################
# Check the seed of all simulation
##############################################################
import helper_funcs as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from HSM import HSM
from visualization import *
from get_host_path import get_host_path
import os
import re
import time
from datetime import datetime
from funcs_for_graphs import *
import param
import scipy.stats as stats
###############################################################
##### Setting
###############################################################
data_dir = "/media/data_cifs/pachaya/FYPsim/HSMmodel/TFtrainingSummary/SciPy_SEEDnumpy/"
TN_filename = lambda seed,trial:"HSMout_theano_SciPytestSeed_Rg1_MaxIter100000_seed%g-%g.npy"%(seed,trial)
TF_filename = lambda seed,trial:"AntolikRegion1_SciPy_jac_npSeed_MaxIter100000_itr1_SEED%g_trial%g_"%(seed,trial)
TN_list_trial = lambda seed:"HSMout_theano_SciPytestSeed_Rg1_MaxIter100000_seed%g-*"%(seed)
TF_list_trial = lambda seed:"AntolikRegion1_SciPy_jac_npSeed_MaxIter100000_itr1_SEED%g_trial*"%(seed)

##############  Download Data Set ############## 
runpath='/home/pachaya/HSMmodel/'
id='1'
rg1_train_input=np.load(runpath+'Data/region'+id+'/training_inputs.npy')
rg1_train_set=np.load(runpath+'Data/region'+id+'/training_set.npy')
raw_vld_set=np.load(runpath+'Data/region'+id+'/raw_validation_set.npy')
rg1_vldinput_set=np.load(runpath+'Data/region'+id+'/validation_inputs.npy')
rg1_vld_set=np.load(runpath+'Data/region'+id+'/validation_set.npy')
(num_im, num_cell) = rg1_train_set.shape
num_seed = 50


# ####### Download trained result ################
Theano_TR_outputs=np.zeros((num_im, num_cell,num_seed))
TensorFlow_TR_outputs=np.zeros((num_im, num_cell,num_seed))
TN_across_seeds ={}
TF_across_seeds ={}
Ks_across_seeds ={}

seed_list = np.arange(50) +1
lgn=9; hlsr=0.2

hsm = HSM(rg1_train_input,rg1_train_set) 
hsm.num_lgn = lgn 
hsm.hlsr = hlsr      
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
    print "------------------------------------------"
    print "    ", ss
    print "Theano"
    #Theano
    fname = TN_filename(ss,0) 
    this_item = fname[:-4] #remove.npy
    tmpitem = np.load(os.path.join(data_dir,fname))
    TN_across_seeds[str(ss)] = tmpitem.item()
    TN_across_seeds[str(ss)]['hsm']=hsm
    Ks_across_seeds[str(ss)]=TN_across_seeds[str(ss)]['x']
    Theano_TR_pred_response[str(ss)] = HSM.response(hsm_tr,rg1_train_input,Ks_across_seeds[str(ss)]) # predicted response after train
    Theano_VLD_pred_response[str(ss)] = HSM.response(hsm_vld,rg1_vldinput_set,Ks_across_seeds[str(ss)]) #predicted response for validation set
    Theano_TR_outputs[:,:,ss-1]=Theano_TR_pred_response[str(ss)]
    print "+++++++"
    print "Tensorflow"
    #Tensorflow
    TF_all_folders = glob.glob(os.path.join(data_dir, TF_filename(ss,0) +'*'))
    for this_item in TF_all_folders: 
        tmpdat=hp.load_TensorFlow_outputs(current_path='', data_dir=data_dir, dir_item=this_item,split_path=False)
        if tmpdat is not None:
            break
    if tmpdat is None:
        print "Error: File not found\t Tensorflow seed = %g"%(ss)
        print this_item
        continue
    TF_across_seeds[str(ss)]  = tmpdat
    TF_TR_pred_response[str(ss)] = TF_across_seeds[str(ss)]['TR_1st_pred_response'] # predicted response after train
    TF_VLD_pred_response[str(ss)] = TF_across_seeds[str(ss)]['VLD_1st_ypredict'] #predicted response for validation set
    TensorFlow_TR_outputs[:,:,ss-1]=TF_TR_pred_response[str(ss)]

tf_sq = np.reshape(TensorFlow_TR_outputs[:,10,:],[1800*50,])
tn_sq = np.reshape(Theano_TR_outputs[:,10,:],[1800*50,])
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.hist(tf_sq,50)
ax1.set_title('TF')
#ax2.hist(np.squeeze(Theano_TR_outputs[:,1,:]), bins=np.arange(0,14,1))
ax2.hist(tn_sq,50)
ax2.set_title('TN')
plt.show()


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.hist(TensorFlow_TR_outputs[10,1,:])
ax1.set_title('TF')
ax2.hist(Theano_TR_outputs[10,1,:])
ax2.set_title('TN')
plt.show()

#Save
np.save('TensorFlow_TR_outputs.npy',TensorFlow_TR_outputs)
np.save('Theano_TR_outputs.npy',Theano_TR_outputs)
##########################################################
## Corr for Theano
import cPickle as pickle
from visualization import *
num_seed=50
num_neuron=103
TN_exp_corr = {}
all_seed=np.arange(num_seed)+1
TN_exp_corr['all_seeds']=all_seed
meanTR=np.zeros([num_seed,num_neuron])
meanVLD=np.zeros([num_seed,num_neuron])
for ss in np.arange(num_seed):
    yhat = Theano_TR_pred_response[str(ss+1)]
    yvld = Theano_VLD_pred_response[str(ss+1)]
    corr = computeCorr(yhat, rg1_train_set)
    corrvld = computeCorr(yvld, rg1_vld_set)
    meanTR[ss,:] = corr
    meanVLD[ss,:] = corrvld
mean_per_seed_TR=  np.mean(meanTR, axis=1)
mean_per_seed_VLD=  np.mean(meanVLD, axis=1)
best_idx=np.argmax(mean_per_seed_VLD)
best_seed = all_seed[best_idx]
print "[VLD]Seed with best average corr coef is %g with R=%f"%(all_seed[np.argmax(mean_per_seed_VLD)],np.max(mean_per_seed_VLD))     
print "[TR] Seed with best average corr coef is %g with R=%f"%(all_seed[np.argmax(mean_per_seed_TR)],np.max(mean_per_seed_TR))     

TN_exp_corr['all_seed']=all_seed
TN_exp_corr['meanTR']=meanTR
TN_exp_corr['meanVLD']=meanVLD
TN_exp_corr['mean_per_seed_TR'] = np.mean(meanTR, axis=1)
TN_exp_corr['mean_per_seed_VLD'] = np.mean(meanVLD, axis=1)

best_seed_TN={}
best_seed_TN['TR corr']=meanTR[best_seed,:]
best_seed_TN['VLD corr']=meanVLD[best_seed,:]
best_seed_TN['results']=None
best_seed_TN['TR_predict']=Theano_TR_pred_response["%g"%best_seed]
best_seed_TN['VLD_predict']=Theano_VLD_pred_response["%g"%best_seed]
SAVE=False
if SAVE:
    data_fold='/home/pachaya/FYP/Analyze/2018-07-25_reviseHSM_antolik/'
    output = open("%sTN_exp_corr.pkl"%(data_fold), 'wb')
    pickle.dump(TN_exp_corr, output)
    output.close()
    output = open("%sbest_seed_TN.pkl"%(data_fold), 'wb')
    pickle.dump(best_seed_TN, output)
    output.close()
    best_seed_results=pickle.load(open("%sbest_seed_results.pkl"%(data_fold), "rb"))
    best_seed_results['Theano']=best_seed_TN
    output = open("%sbest_seed_results_w_TN.pkl"%(data_fold), 'wb')
    pickle.dump(best_seed_results, output)
    output.close()
##########################################################

#Load
TensorFlow_TR_outputs=np.load('TensorFlow_TR_outputs.npy')
Theano_TR_outputs=np.load('Theano_TR_outputs.npy')
num_im=1800; num_cell=103; num_seed=50
TF_seed_stats=np.zeros((num_cell,num_im,2))
TN_seed_stats=np.zeros((num_cell,num_im,2))
stats_ind=np.zeros((num_cell,num_im,2))
stats_1samp=np.zeros((num_cell,num_im,2))
p_ind=np.zeros((num_cell,num_im))
p_1samp=np.zeros((num_cell,num_im))
for cc in np.arange(num_cell):
    print "cc = ",cc
    for ii in np.arange(num_im):
        print "ii = ", ii
        tf_act = TensorFlow_TR_outputs[ii,cc,:]
        tn_act = Theano_TR_outputs[ii,cc,:]
        TF_seed_stats[cc,ii,:]=[tf_act.mean(),tf_act.var()]
        TN_seed_stats[cc,ii,:]=[tn_act.mean(),tn_act.var()]
        t_i, p_i = stats.ttest_ind(tn_act,tf_act)
        t1, p1 = stats.ttest_1samp(tf_act, popmean=tn_act.mean())
        stats_ind[cc,ii,:]=[t_i,p_i]
        stats_1samp[cc,ii,:]=[t1,p1]
        p_ind[cc,ii]=p_i; p_1samp[cc,ii]=p1
    print "-----------------------------"


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.hist(tf_act,10)
ax1.set_title('TF')
ax2.hist(tn_act,10)
ax2.set_title('TN')
plt.show()

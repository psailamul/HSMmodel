"""
All analysis
"""
import matplotlib.pyplot as plt
import funcs_for_graphs as hp #Helper functions
import funcs_for_graphs as vis #Helper functions
from datetime import datetime
from scipy import stats
from HSM import HSM
import numpy as np
import param
import glob
import os
import re
import time
import scipy.stats as stats

########################################################################
# Setting
########################################################################
SAVEFIG=False
dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
hsm = hp.build_hsm_for_Theano()

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
NUM_SEEDS = 50
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
        tmpdat=hp.load_TensorFlow_outputs('', data_dir, this_item,split_path=False)
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
modelbuilding_time = time.time() 
for ss in seed_list:
    print ss
    #Theano
    Theano_TR_pred_response[str(ss)] = HSM.response(hsm_tr,rg1_train_input,Ks_across_seeds[str(ss)]) # predicted response after train
    Theano_VLD_pred_response[str(ss)] = HSM.response(hsm_vld,rg1_vldinput_set,Ks_across_seeds[str(ss)]) #predicted response for validation set
    #TensorFlow
    TF_TR_pred_response[str(ss)] = TF_across_seeds[str(ss)]['TR_1st_pred_response'] # predicted response after train
    TF_VLD_pred_response[str(ss)] = TF_across_seeds[str(ss)]['VLD_1st_ypredict'] #predicted response for validation set
print "Building model/Get predicted response complete: Time %s" %(time.time() - modelbuilding_time)

# #############  Mean activity to validation set ############## 
TN_corr_seeds=np.zeros([50,103]); 
TN_vld_corr_seeds=np.zeros([50,103])
TF_corr_seeds=np.zeros([50,103]); 
TF_vld_corr_seeds=np.zeros([50,103])
ttest_result_TR=[]
pval_TR=np.zeros(50)
ttest_result_VLD=[]
pval_VLD=np.zeros(50)
pval_direct_TR=np.zeros(50)
pval_direct_VLD=np.zeros(50)
for ss in seed_list:
    id=str(ss)
    print "id: ", id
    TN_corr_seeds[ss-1,:] = hp.computeCorr(Theano_TR_pred_response[id],rg1_train_set) #corr per neuron
    TN_vld_corr_seeds[ss-1,:]=hp.computeCorr(Theano_VLD_pred_response[id],rg1_vld_set)
    TF_corr_seeds[ss-1,:] = hp.computeCorr(TF_TR_pred_response[id],rg1_train_set)
    TF_vld_corr_seeds[ss-1,:]=hp.computeCorr(TF_VLD_pred_response[id],rg1_vld_set)   
    ttestTR= stats.ttest_ind(TN_corr_seeds[ss-1,:],TF_corr_seeds[ss-1,:])
    ttest_result_TR.append(ttestTR)
    pval_TR[ss-1] = ttestTR.pvalue
    ttestVLD= stats.ttest_ind(TN_vld_corr_seeds[ss-1,:],TF_vld_corr_seeds[ss-1,:])
    ttest_result_VLD.append(ttestVLD)
    pval_VLD[ss-1]= ttestVLD.pvalue
    tmpTR= stats.ttest_rel(Theano_TR_pred_response[id].flatten(),TF_TR_pred_response[id].flatten())
    pval_direct_TR[ss-1]=tmpTR.pvalue
    tmpVLD= stats.ttest_rel(Theano_VLD_pred_response[id].flatten(),TF_VLD_pred_response[id].flatten())
    pval_direct_VLD[ss-1]=tmpVLD.pvalue
resTR=stats.ttest_rel(TN_corr_seeds,TF_corr_seeds) #Cell ID 67 - significantly difference
resVLD=stats.ttest_rel(TN_vld_corr_seeds,TF_vld_corr_seeds)

best_seed_TN =np.argmax(np.mean(TN_corr_seeds,axis=1))
print("Best seed of TN on training set is #%g with mean corr = %g"%(best_seed_TN,np.max(np.mean(TN_corr_seeds,axis=1))))
best_seed_TF =np.argmax(np.mean(TF_corr_seeds,axis=1))
print("Best seed of TF on training set is #%g with mean corr = %g"%(best_seed_TF,np.max(np.mean(TF_corr_seeds,axis=1))))


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
    r_sqr = hp.TN_TF_Rsquare(response1, response2)
    R2_training.append(r_sqr)
    #Test set
    setname ='TEST'
    response1 =Theano_VLD_pred_response[id]
    response2 =TF_VLD_pred_response[id]
    r_sqr = hp.TN_TF_Rsquare(response1, response2)
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
    ttestTR= stats.ttest_rel(Theano_TR_pred_response[id].flatten(),TF_TR_pred_response[id].flatten())
    ttest_result_TR.append(ttestTR)
    pval_TR[ss-1] = ttestTR.pvalue
    ttestVLD= stats.ttest_rel(Theano_VLD_pred_response[id].flatten(),TF_VLD_pred_response[id].flatten())
    ttest_result_VLD.append(ttestVLD)
    pval_VLD[ss-1]= ttestVLD.pvalue
#============================================================================

# #############  T-Test by condition ############## 

TR_across_seeds_ttest = []
TR_across_seeds_pval = np.zeros([1800*103])
VLD_across_seeds_ttest =[]
VLD_across_seeds_pval = np.zeros([50*103])

for ci in range(1800*103):    
    tmpttest = stats.ttest_ind(allseeds_TN_TR[:,ci],allseeds_TF_TR[:,ci])
    TR_across_seeds_ttest.append(tmpttest)
    TR_across_seeds_pval[ci] = tmpttest.pvalue

for ci in range(50*103):    
    tmpttest = stats.ttest_ind(allseeds_TN_VLD[:,ci],allseeds_TF_VLD[:,ci])
    VLD_across_seeds_ttest.append(tmpttest)
    VLD_across_seeds_pval[ci] = tmpttest.pvalue    
print("Number of VLD_across_seeds_pval < 0.001 : %g"%(np.sum(VLD_across_seeds_pval<0.001)))
#============================================================================

# #############  mean difference square VS corr coef of TF (per neuron, all seeds) ############## 

#Calculate     sum(TNi - TFi)^2  by neuron 
# (TF_vld_corr_seeds)  (#seed , #neuron) = (50,103)
#  onebig_TN_VLD, onebig_TF_VLD  #3D: Image x neuron x seed  = (50,103,50)
L2 = np.zeros(TF_vld_corr_seeds.shape)
for si in range(NUM_SEEDS):
	tmpTN =  onebig_TN_VLD[:,:,si]
	tmpTF =  onebig_TF_VLD[:,:,si]
	diff = tmpTN-tmpTF
	diffsq = np.multiply(diff,diff)
	tmpL2 = np.mean(diffsq,axis=0)
	L2[si,:] = tmpL2
check_corr = stats.pearsonr(L2.flatten(), TF_vld_corr_seeds.flatten())
plt.figure()
plt.scatter(L2.flatten(), TF_vld_corr_seeds.flatten())
plt.xlabel('Mean difference square')
plt.ylabel('Correlation Coefficients of Validation Set')
plt.title("Correlation coefficients = %g with p=%.4f"%(check_corr[0], check_corr[1]))

check_corr2 = stats.pearsonr(L2[best_seed_TF,:],TF_vld_corr_seeds[best_seed_TF,:])
plt.figure()
plt.scatter(L2[best_seed_TF,:],TF_vld_corr_seeds[best_seed_TF,:])
plt.xlabel('MSE')
plt.ylabel('Correlation Coefficients of validation set')
plt.title("Best seed :Corr of MSE to performance of validation set = %g with p=%.4f"%(check_corr2[0], check_corr2[1]))

plt.show()
#Mismatch between TN,TF doesn't corr with resulted error

#============================================================================

# ############## Histogram of VLD ############## 

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
num_bins = 100

fig, ax = plt.subplots()

# the histogram of the data
weights = np.ones_like(allseeds_TN_VLD.flatten())*100.0/float(len(allseeds_TN_VLD.flatten()))
bins = np.linspace(0, 10, 50)
n1, bins1, patches1 = ax.hist(allseeds_TN_VLD.flatten(), bins, alpha=0.5, label='Theano', normed=1, weights=weights)
n2, bins2, patches2 = ax.hist(allseeds_TF_VLD.flatten(), bins, alpha=0.5, label='TensorFlow', normed=1, weights=weights)
w2 = np.ones_like(rg1_vld_set.flatten())*100.0/float(len(rg1_vld_set.flatten()))
n3, bins3, patches3 = ax.hist(rg1_vld_set.flatten(), bins, alpha=0.5, label='Record', normed=1, weights=w2)
plt.legend(loc='upper right')
ax.set_xlabel('Neural activity (spikes)')
ax.set_ylabel('Density(%)')
ax.set_title(r'Distribution of predicted and recorded neural response')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

#############################
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
num_bins = 100

fig, ax = plt.subplots()

# the histogram of the data
weights = np.ones_like(allseeds_TN_VLD.flatten())*100.0/float(len(allseeds_TN_VLD.flatten()))
bins = np.linspace(0,10, 200)
n1, bins1, patches1 = ax.hist(allseeds_TN_VLD.flatten(), bins, alpha=0.5, label='Theano', weights=weights)
n2, bins2, patches2 = ax.hist(allseeds_TF_VLD.flatten(), bins, alpha=0.5, label='TensorFlow',weights=weights)
w2 = np.ones_like(rg1_vld_set.flatten())*100.0/float(len(rg1_vld_set.flatten()))
n3, bins3, patches3 = ax.hist(rg1_vld_set.flatten(), bins, alpha=0.5, label='Recorded', weights=w2)
plt.legend(loc='upper right')
ax.set_xlim(xmin=0.0, xmax=2)
ax.set_xlabel('Neural activity (spikes)')
ax.set_ylabel('Density(%)')
ax.set_title(r'Distribution of predicted and recorded neural response')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

################################

# Box plot 
f, (ax1, ax2,ax3) = plt.subplots(3, 1, sharey=True)

ax1.boxplot(allseeds_TN_VLD.T, 0, '')
ax1.set_title('TN')

ax2.boxplot(allseeds_TF_VLD.T, 0, '')
ax2.set_title('TF')

ax3.boxplot(rg1_vld_set.flatten(), 0, '')
ax3.set_title('recorded')

plt.show()
#============================================================================

# ############# Example predicted VLD response TN-TF for best TF seed  ############## 
# Use best_seed_TF
# Pick best neuron from corr(TF, Recorded) -- plot activity 3 items TN TF recorded, report mean diff square
# Pick median neuron from corr(TF, Recorded)
# Distribution of TNTF corr?
# Rank neurons based on mean diff square? 

# (TF_vld_corr_seeds)  (#seed , #neuron) = (50,103)
#  onebig_TN_VLD, onebig_TF_VLD  #3D: Image x neuron x seed  = (50,103,50)
TF_VLD_bestTFseeds = onebig_TF_VLD[:,:,best_seed_TF]
TN_VLD_bestTFseeds = onebig_TN_VLD[:,:,best_seed_TF]
TF_vld_corr_bestTFseeds = vis.computeCorr(TF_VLD_bestTFseeds,rg1_vld_set)
TN_vld_corr_bestTFseeds = vis.computeCorr(TN_VLD_bestTFseeds,rg1_vld_set) 

stats_param ='max'
imax = np.argmax(TF_vld_corr_bestTFseeds)

fig, ax = plt.subplots(figsize=[16,4])
plt.subplot(1,2,1)
plt.plot(rg1_vld_set[:,imax],'-kp',label='Measured Response')
plt.plot(TF_VLD_bestTFseeds[:, imax],'--or',label=r'HSM$_{TF}$ Response R=%f'%(TF_vld_corr_bestTFseeds[imax]))
plt.plot(TN_VLD_bestTFseeds[:, imax],'--ob',label=r'HSM$_{TN}$ Response R=%f'%(TN_vld_corr_bestTFseeds[imax]))
plt.ylabel('Response')
plt.xlabel('Image #')
plt.title("Cell#%g is the %s  neuron, R = %.5f, mean neuron of TF has R = %.5f"%(imax+1 ,
                                                                                 stats_param,
                                                                                 np.max(TF_vld_corr_bestTFseeds), 
                                                                                 np.mean(TF_vld_corr_bestTFseeds)))
plt.legend(loc=0)

plt.subplot(1,2,2)
plt.scatter(rg1_vld_set[:,imax], TF_VLD_bestTFseeds[:,imax],color='r',label=r'HSM$_{TF}$')
plt.scatter(rg1_vld_set[:,imax], TN_VLD_bestTFseeds[:,imax],color='b',label=r'HSM$_{TN}$')
N=np.ceil(np.max([np.max(TF_VLD_bestTFseeds[:,imax]),np.max(TN_VLD_bestTFseeds[:,imax]),np.max(rg1_vld_set[:,imax])]))
plt.plot(np.arange(N),np.arange(N),'--c')
plt.xlim([0,N]); plt.ylim([0,N])
plt.ylabel('Predicted Response')
plt.xlabel('Measured Response')
plt.title('Scatter plot of measured response and predicted response of cell#%g'%(imax+1)) 

plt.show()

stats_param ='median'
N = len(TF_vld_corr_bestTFseeds)
medidx = N/2-1 if N%2==0 else (N-1)/2
idx = np.argsort(TF_vld_corr_bestTFseeds)[medidx]
med_corr = TF_vld_corr_bestTFseeds[idx]

fig, ax = plt.subplots(figsize=[16,4])
plt.subplot(1,2,1)
plt.plot(rg1_vld_set[:,idx],'-kp',label='Measured Response')
plt.plot(TF_VLD_bestTFseeds[:, idx],'--or',label=r'HSM$_{TF}$ Response R=%f'%(TF_vld_corr_bestTFseeds[idx]))
plt.plot(TN_VLD_bestTFseeds[:, idx],'--ob',label=r'HSM$_{TN}$ Response R=%f'%(TN_vld_corr_bestTFseeds[idx]))
plt.ylabel('Response')
plt.xlabel('Image #')
plt.title("Cell#%g is the %s  neuron, R = %.5f, mean neuron of TF has R = %.5f"%(idx+1 ,
                                                                                 stats_param,
                                                                                 np.median(TF_vld_corr_bestTFseeds), 
                                                                                 np.mean(TF_vld_corr_bestTFseeds)))
plt.legend(loc=0)

plt.subplot(1,2,2)
plt.scatter(rg1_vld_set[:,idx], TF_VLD_bestTFseeds[:,idx],color='r',label=r'HSM$_{TF}$')
plt.scatter(rg1_vld_set[:,idx], TN_VLD_bestTFseeds[:,idx],color='b',label=r'HSM$_{TN}$')
N=np.ceil(np.max([np.max(TF_VLD_bestTFseeds[:,idx]),np.max(TN_VLD_bestTFseeds[:,idx]),np.max(rg1_vld_set[:,idx])]))
plt.plot(np.arange(N),np.arange(N),'--c')
plt.xlim([0,N]); plt.ylim([0,N])
plt.ylabel('Predicted Response')
plt.xlabel('Measured Response')
plt.title('Scatter plot of measured response and predicted response of cell#%g'%(idx+1)) 
 
plt.show()

# Stats of TN for comparison

print("Best seed of TN on training set is #%g with stats for VLD set"%(best_seed_TN))
print("MEAN: %g, max: %g, median: %g\n"%(np.mean(TN_vld_corr_seeds[best_seed_TN,:]),
                                           np.max(TN_vld_corr_seeds[best_seed_TN,:]),
                                           np.median(TN_vld_corr_seeds[best_seed_TN,:])))
print("----------------------------------------------------------------------------------------")
print("Best seed of TF on training set is #%g with stats for VLD set"%(best_seed_TF))
print("MEAN: %g, max: %g, median: %g\n"%(np.mean(TF_vld_corr_seeds[best_seed_TF,:]),
                                           np.max(TF_vld_corr_seeds[best_seed_TF,:]),
                                           np.median(TF_vld_corr_seeds[best_seed_TF,:])))


# Box plot 

boxplot_VLD_bestseed = np.zeros([2,rg1_vld_set.shape[1]])
boxplot_VLD_bestseed[0,:] = TN_vld_corr_seeds[best_seed_TN,:]
boxplot_VLD_bestseed[1,:] = TF_vld_corr_seeds[best_seed_TF,:]

plt.figure()
plt.boxplot(boxplot_VLD_bestseed.T,1,'o')
plt.xticks([1, 2], ['TN', 'TF']) 
plt.ylabel('R')
plt.xlabel('Models')
plt.show()

#============================================================================

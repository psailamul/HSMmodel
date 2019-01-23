# Plot figures for the report
import funcs_for_graphs as vis #Helper functions
import helper_funcs as hp #Helper functions
from datetime import datetime
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

# ############## Specified Folder ##########################
DATA_LOC = '/media/data_cifs/pachaya/FYPsim/HSMmodel/TFtrainingSummary/'
data_dir =  os.path.join(DATA_LOC,'SciPy_SEEDnumpy')
FIGURES_LOC = os.path.join(os.getcwd(), 'Figures','figs_for_report')
figures_dir = os.path.join(FIGURES_LOC,"%s"%(dt_stamp))
if SAVEFIG:
    if not os.path.isdir(figures_dir) :
        os.mkdir(figures_dir)

FIG_HANDLES=[]
FIG_NAMES=[]
        
TN_filename = lambda rg,seed,trial:"HSMout_theano_SciPytestSeed_Rg%g_MaxIter100000_seed%g-%g.npy"%(rg,seed,trial)
TF_filename = lambda rg,seed,trial:"AntolikRegion%g_SciPy_jac_npSeed_MaxIter100000_itr1_SEED%g_trial%g_"%(rg,seed,trial)


########################################################################
# Fig 4 for tfHSM all regions
########################################################################

SEED=13
TRIAL=0
LGN=9
HLSR=0.2
# #############  Download Data Set ############## 
NUM_REGIONS =3
train_input ={}; train_set={}; raw_vld_set={}; vldinput_set={}; vld_set={}
runpath = os.getcwd() +'/'

for i in range(NUM_REGIONS):
    id=str(i+1)
    train_input[id]=np.load(runpath+'Data/region'+id+'/training_inputs.npy')
    train_set[id]=np.load(runpath+'Data/region'+id+'/training_set.npy')
    raw_vld_set[id]=np.load(runpath+'Data/region'+id+'/raw_validation_set.npy')
    vldinput_set[id]=np.load(runpath+'Data/region'+id+'/validation_inputs.npy')
    vld_set[id]=np.load(runpath+'Data/region'+id+'/validation_set.npy')

# ####### Download VLD result ################
Theano_outputs={}; TensorFlow_outputs={}
for i in range(NUM_REGIONS):
    Theano_outputs[str(i+1)]=None
    TensorFlow_outputs[str(i+1)]=None

Ks={}; TN =Theano_outputs; TF=TensorFlow_outputs;

for i in range(NUM_REGIONS):
    id=str(i+1)
    hsm = HSM(vldinput_set[id],vld_set[id]) # Initialize model --> add input and output, construct parameters , build mobel, # create loss function
    print "Created HSM model"   
    hsm.num_lgn = LGN
    hsm.hlsr = HLSR

    #Theano
    fname = TN_filename(i+1, SEED, TRIAL) 
    this_item = fname[:-4] #remove.npy
    tmpitem = np.load(os.path.join(data_dir,fname))
    TN[id] = tmpitem.item()
    TN[id]['hsm']=hsm
    Ks[id]=TN[id]['x']
    
    #Tensorflow
    TF_all_folders = glob.glob(os.path.join(data_dir, TF_filename(i+1, SEED, TRIAL) +'*'))
    for this_item in TF_all_folders: 
        tmpdat=hp.load_TensorFlow_outputs('', data_dir, this_item,split_path=False)
        if tmpdat is not None:
            break
    if tmpdat is None:
        print "Error: File not found\t"
        print this_item
    TF[id]  = tmpdat
    tmpdat = None

# #############  Get predicted response ############## 
Theano_VLD_pred_response={};
TF_VLD_pred_response={}

for i in range(NUM_REGIONS):
    id=str(i+1)
    #Theano
    Theano_VLD_pred_response[id] = HSM.response(TN[id]['hsm'],vldinput_set[id],Ks[id]) #predicted response for validation set
    #TensorFlow
    TF_VLD_pred_response[id] = TensorFlow_outputs[id]['VLD_1st_ypredict'] #predicted response for validation set

# ############# Calculate correlation coefficient ############## 
TN_corr={}; TN_vld_corr={}
TF_corr={}; TF_vld_corr={}
TNTF_corr={}; TNTF_vld_corr={}
combined_TNvld_corr =[]
combined_TFvld_corr =[]

for i in range(NUM_REGIONS):
    id=str(i+1)
    key = id
    TN_vld_corr[id]=vis.computeCorr(Theano_VLD_pred_response[id],vld_set[id])
    TF_vld_corr[id]=vis.computeCorr(TF_VLD_pred_response[id],vld_set[id])
    TNTF_vld_corr[id]=vis.computeCorr(Theano_VLD_pred_response[id],TF_VLD_pred_response[id])
    TNvld = TN_vld_corr[key]; combined_TNvld_corr.append(TNvld)
    TFvld = TF_vld_corr[key]; combined_TFvld_corr.append(TFvld)
    print("[Theano] Region%s: mean=%g, max=%g, median=%g, min=%g"%(key,np.mean(TNvld),np.max(TNvld), np.median(TNvld), np.min(TNvld)))
    print("[TensorFlow] Region%s: mean=%g, max=%g, median=%g, min=%g"%(key,np.mean(TFvld),np.max(TFvld), np.median(TFvld), np.min(TFvld)))
    print("-----------------------------------------------")
combined_TNvld_corr = np.concatenate(combined_TNvld_corr)
combined_TFvld_corr = np.concatenate(combined_TFvld_corr)
print("-----------------------------------------------")
print("\t COMBINE ALL 3 REGIONS")
print("-----------------------------------------------")
print("Theano")
print("\t mean=%g \n\t max=%g \n\t median=%g \n\t min=%g"%(np.mean(combined_TNvld_corr),
                                                            np.max(combined_TNvld_corr), 
                                                            np.median(combined_TNvld_corr), 
                                                            np.min(combined_TNvld_corr)))
print("-----------------------------------------------")
print("TensorFlow")
print("\t mean=%g \n\t max=%g \n\t median=%g \n\t min=%g"%(np.mean(combined_TFvld_corr),
                                                            np.max(combined_TFvld_corr), 
                                                            np.median(combined_TFvld_corr), 
                                                            np.min(combined_TFvld_corr)))
print("-----------------------------------------------")

# ###################### Plot Sample Result #################
#Theano : Validation Set
fig=vis.compare_corr_all_regions(Theano_VLD_pred_response,vld_set, TN_vld_corr, stats_param='max', titletxt='Theano :: Validation Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_TN_VLD_fig_max')
fig=vis.compare_corr_all_regions(Theano_VLD_pred_response,vld_set, TN_vld_corr, stats_param='median', titletxt='Theano :: Validation Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_TN_VLD_fig_med')

# TensorFlow : Validation Set
fig=vis.compare_corr_all_regions(TF_VLD_pred_response,vld_set, TF_vld_corr, stats_param='max', titletxt='TensorFlow :: Validation Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_TF_VLD_fig_max')
fig=vis.compare_corr_all_regions(TF_VLD_pred_response,vld_set, TF_vld_corr, stats_param='median', titletxt='TensorFlow :: Validation Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_TF_VLD_fig_med')

# Theano & TensorFlow

fig=vis.compare_corr_all_regions(Theano_VLD_pred_response,TF_VLD_pred_response, TNTF_vld_corr, stats_param='max', titletxt='Theano&TensorFlow :: Validation Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('Z0_TNTF_VLD_fig_max')

# #################################################
# Plot in Figure 4
# #################################################


# Fig 4A : all regions
titletxt='Validation Set in all regions'
#TN
# # Max
fig=vis.plot_corr_response_scatter(pred_response=vis.combine_responses(Theano_VLD_pred_response), 
    vld_set=vis.combine_responses(vld_set), 
    corr_set=vis.combine_corrs(TN_vld_corr), 
    stats_param='max',
    titletxt='Theano : '+titletxt, 
    RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('A_combine_TN_vld_sample_max')

# # Median
fig=vis.plot_corr_response_scatter(pred_response=vis.combine_responses(Theano_VLD_pred_response), 
    vld_set=vis.combine_responses(vld_set), 
    corr_set=vis.combine_corrs(TN_vld_corr), 
    stats_param='median',
    titletxt='Theano : '+titletxt, 
    RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('B_combine_TN_vld_sample_median')


#TF
# # Max
fig=vis.plot_corr_response_scatter(pred_response=vis.combine_responses(TF_VLD_pred_response), 
    vld_set=vis.combine_responses(vld_set), 
    corr_set=vis.combine_corrs(TF_vld_corr), 
    stats_param='max',
    titletxt='TensorFlow : '+titletxt, 
    RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('A_combine_TF_vld_sample_max')

# # Median
fig=vis.plot_corr_response_scatter(pred_response=vis.combine_responses(TF_VLD_pred_response), 
    vld_set=vis.combine_responses(vld_set), 
    corr_set=vis.combine_corrs(TF_vld_corr), 
    stats_param='median',
    titletxt='TensorFlow : '+titletxt, 
    RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('B_combine_TF_vld_sample_median')

# FIG 4 C : CDF

fig,Xs,Fs = vis.cdf_allregions( TN_vld_corr, NUM_REGIONS=3, DType='Theano Validation set : ', C_CODE=True, SHOW=True, RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('C_TN_VLD_CDF')

fig,Xs,Fs = vis.cdf_allregions( TF_vld_corr, NUM_REGIONS=3, DType='TensorFlow Validation set : ', C_CODE=True, SHOW=True, RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('C_TF_VLD_CDF')

fig,Xs,Fs = vis.cdf_allregions( TNTF_vld_corr, NUM_REGIONS=3, DType='Validation set of Theano and TensorFlow version: ', C_CODE=True, SHOW=True, RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('C_TNTF_VLD_CDF')



# #################################################
# Save all figures
# #################################################

if(SAVEFIG):
    for fignum in range(len(FIG_HANDLES)):
        fhandle =FIG_HANDLES[fignum]
        fname=FIG_NAMES[fignum]
        fhandle.savefig(os.path.join(figures_dir,fname+'.png'))

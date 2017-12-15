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
"""    
# Pseudo code for plotting 
1. Plot y_pred of Theano and TF 
2. Fit Linear Regression (?) or --- the model is Y = X  --> MSE and R^2 

3. Calculate R^2 
4. Plot FEV Theano and TF


-----
 mean squared error (MSE) or the sum of squares of error (SSE), also called the “residual sum of squares.” (RSS)
----
The definition of R-squared is fairly straight-forward; it is the percentage of the response variable variation that is explained by a linear model. Or:

R-squared = Explained variation / Total variation
----

then the variability of the data set can be measured using three sums of squares formulas:
The total sum of squares (proportional to the variance of the data):
{\displaystyle SS_{\text{tot}}=\sum _{i}(y_{i}-{\bar {y}})^{2},} SS_{\text{tot}}=\sum _{i}(y_{i}-{\bar {y}})^{2},
The regression sum of squares, also called the explained sum of squares:
{\displaystyle SS_{\text{reg}}=\sum _{i}(f_{i}-{\bar {y}})^{2},} SS_{\text{reg}}=\sum _{i}(f_{i}-{\bar {y}})^{2},
The sum of squares of residuals, also called the residual sum of squares:
{\displaystyle SS_{\text{res}}=\sum _{i}(y_{i}-f_{i})^{2}=\sum _{i}e_{i}^{2}\,} {\displaystyle SS_{\text{res}}=\sum _{i}(y_{i}-f_{i})^{2}=\sum _{i}e_{i}^{2}\,}
The most general definition of the coefficient of determination is
{\displaystyle R^{2}\equiv 1-{SS_{\rm {res}} \over SS_{\rm {tot}}}.\,} R^{2}\equiv 1-{SS_{\rm {res}} \over SS_{\rm {tot}}}.\,

https://en.wikipedia.org/wiki/Coefficient_of_determination
-----=


ei = yi - fi
ybar = mean
SStot = sum (yi-ymean)^2

"explained sum of square"
SSreg = sum (fi - ybar)^2

"Residual sum of square"
SSres = sum (yi - fi)^2 = sum ei^2

R^2 = 1 - SSres / SStotal


"Fraction of variance unexplained"
R^2 = 1-FVU
FVU = 1-R^2

FVU = Varerr / VARtotal = (SSerr/n ) / (SStot/n) = SSerr / SStot = 1 - SSreg/SStot (for linear regression)

SSerr = sum (yi - fi)^2

or FVU = MSE(f) / var(Y) 
https://en.wikipedia.org/wiki/Fraction_of_variance_unexplained


--



"""
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
Code='TN_TF_comparison'
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

# ############# Calculate correlation coefficient ############## 
TN_corr={}; TN_vld_corr={}
TF_corr={}; TF_vld_corr={}
TNTF_corr={}; TNTF_vld_corr={}

for i in range(NUM_REGIONS):
    id=str(i+1)
    TN_corr[id] = computeCorr(Theano_TR_pred_response[id],train_set[id])
    TN_vld_corr[id]=computeCorr(Theano_VLD_pred_response[id],vld_set[id])
    TF_corr[id] = computeCorr(TF_TR_pred_response[id],train_set[id])
    TF_vld_corr[id]=computeCorr(TF_VLD_pred_response[id],vld_set[id])
    TNTF_corr[id] = computeCorr(Theano_TR_pred_response[id],TF_TR_pred_response[id])
    TNTF_vld_corr[id]=computeCorr(Theano_VLD_pred_response[id],TF_VLD_pred_response[id])

# ###################### Plot Sample Result #################
#Theano : Training Set
fig=compare_corr_all_regions(Theano_TR_pred_response,train_set, TN_corr, stats_param='max', titletxt='Theano :: Training Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_TN_TR_fig_max')
fig=compare_corr_all_regions(Theano_TR_pred_response,train_set, TN_corr, stats_param='median', titletxt='Theano :: Training Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_Tn_TR_fig_med')

#Theano : Validation Set
fig=compare_corr_all_regions(Theano_VLD_pred_response,vld_set, TN_vld_corr, stats_param='max', titletxt='Theano :: Validation Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_TN_VLD_fig_max')
fig=compare_corr_all_regions(Theano_VLD_pred_response,vld_set, TN_vld_corr, stats_param='median', titletxt='Theano :: Validation Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_TN_VLD_fig_med')

#TensorFlow : Training Set
fig=compare_corr_all_regions(TF_TR_pred_response,train_set, TF_corr, stats_param='max', titletxt='TF:: 1st Training Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_TF_TR_fig_max')
fig=compare_corr_all_regions(TF_TR_pred_response,train_set, TF_corr, stats_param='median', titletxt='TF:: 1st Training Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_TF_TR_fig_med')

#TensorFlow : Validation Set
fig=compare_corr_all_regions(TF_VLD_pred_response,vld_set, TF_vld_corr, stats_param='max', titletxt='TF :: 1st Validation Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_TF_VLD_fig_max')
fig=compare_corr_all_regions(TF_VLD_pred_response,vld_set, TF_vld_corr, stats_param='median', titletxt='TF:: 1st Validation Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('0_TF_VLD_fig_med')

# Theano & TensorFlow
fig=compare_corr_all_regions(Theano_TR_pred_response,TF_TR_pred_response, TNTF_corr, stats_param='max', titletxt='Theano&TensorFlow :: Training Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('Z0_TNTF_TR_fig_max')
fig=compare_corr_all_regions(Theano_VLD_pred_response,TF_VLD_pred_response, TNTF_vld_corr, stats_param='max', titletxt='Theano&TensorFlow :: Validation Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('Z0_TNTF_VLD_fig_max')

 # #################################################
# Plot in Figure 4
# #################################################


# Fig 4A : all regions
titletxt='Validation Set in all regions'
#TN
# # Max
fig=plot_corr_response_scatter(pred_response=combine_responses(Theano_VLD_pred_response), 
    vld_set=combine_responses(vld_set), 
    corr_set=combine_corrs(TN_vld_corr), 
    stats_param='max',
    titletxt='Theano : '+titletxt, 
    RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('A_combine_TN_vld_sample_max')

# # Median
fig=plot_corr_response_scatter(pred_response=combine_responses(Theano_VLD_pred_response), 
    vld_set=combine_responses(vld_set), 
    corr_set=combine_corrs(TN_vld_corr), 
    stats_param='median',
    titletxt='Theano : '+titletxt, 
    RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('B_combine_TN_vld_sample_median')


#TF
# # Max
fig=plot_corr_response_scatter(pred_response=combine_responses(TF_VLD_pred_response), 
    vld_set=combine_responses(vld_set), 
    corr_set=combine_corrs(TF_vld_corr), 
    stats_param='max',
    titletxt='TensorFlow : '+titletxt, 
    RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('A_combine_TF_vld_sample_max')

# # Median
fig=plot_corr_response_scatter(pred_response=combine_responses(TF_VLD_pred_response), 
    vld_set=combine_responses(vld_set), 
    corr_set=combine_corrs(TF_vld_corr), 
    stats_param='median',
    titletxt='TensorFlow : '+titletxt, 
    RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('B_combine_TF_vld_sample_median')


#TN&TF
titletxt='Compare validation set of Theano & TensorFlow'
# # Max
fig=plot_corr_response_scatter(pred_response=combine_responses(TF_VLD_pred_response), 
    vld_set=combine_responses(Theano_VLD_pred_response), 
    corr_set=combine_corrs(TNTF_vld_corr), 
    stats_param='max',
    titletxt=titletxt, 
    RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('ZA_combine_TNTF_vld_sample_max')

fig=plot_corr_response_scatter(pred_response=combine_responses(TF_VLD_pred_response), 
    vld_set=combine_responses(Theano_VLD_pred_response), 
    corr_set=combine_corrs(TNTF_vld_corr), 
    stats_param='med',
    titletxt=titletxt, 
    RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('ZA_combine_TNTF_vld_sample_med')


#Histogram of TN & TF correlation
combine_TNTF_corr = combine_corrs(TNTF_corr)
combine_TNTF_vld_corr =  combine_corrs(TNTF_vld_corr)
fig=plt.figure()
plt.subplot(121); plt.hist(combine_TNTF_corr,normed=True); plt.xlim([0,1]); plt.title('Training set')
plt.subplot(122); plt.hist(combine_TNTF_vld_corr,normed=True); plt.xlim([0,1]); plt.title('Validation set')
plt.suptitle('Distribution of correlation coefficient between response from Theano and TensorFlow')
plt.show()
FIG_HANDLES.append(fig); FIG_NAMES.append('Z_combine_TNTF_histogram')

# FIG 4 C : CDF

fig,Xs,Fs = cdf_allregions( TN_corr, NUM_REGIONS=3, DType='Theano Training set : ', C_CODE=True, SHOW=True, RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('C_TN_TR_CDF')
fig,Xs,Fs = cdf_allregions( TN_vld_corr, NUM_REGIONS=3, DType='Theano Validation set : ', C_CODE=True, SHOW=True, RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('C_TN_VLD_CDF')

fig,Xs,Fs = cdf_allregions( TF_corr, NUM_REGIONS=3, DType='TensorFlow Training set : ', C_CODE=True, SHOW=True, RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('C_TF_TR_CDF')
fig,Xs,Fs = cdf_allregions( TF_vld_corr, NUM_REGIONS=3, DType='TensorFlow Validation set : ', C_CODE=True, SHOW=True, RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('C_TF_VLD_CDF')

fig,Xs,Fs = cdf_allregions( TNTF_corr, NUM_REGIONS=3, DType='Training set of Theano and TensorFlow version: ', C_CODE=True, SHOW=True, RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('C_TNTF_TR_CDF')
fig,Xs,Fs = cdf_allregions( TNTF_vld_corr, NUM_REGIONS=3, DType='Validation set of Theano and TensorFlow version: ', C_CODE=True, SHOW=True, RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('C_TNTF_VLD_CDF')





if(SAVEFIG):
    for fignum in range(len(FIG_HANDLES)):
        fhandle =FIG_HANDLES[fignum]
        fname=FIG_NAMES[fignum]
        fhandle.savefig(Fig_fold+fname+'.png')

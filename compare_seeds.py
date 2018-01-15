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
import glob
import param
from scipy.stats import ttest_rel

# # ############# Functions ###############
def get_param_from_fname(fname, keyword):
    cuts = re.split('_',fname)
    for prm in cuts:
        if str.startswith(prm,keyword):
            return prm[len(keyword):]
    else:
        print "WARNING:: KEYWORD NOT FOUND"
        print "Keyword = %s"%(keyword)
        return None
    if not str.endswith(dir_item,'/'):
        dir_item = "%s/"%(dir_item)
    directory = current_path+data_dir+dir_item
    
def load_TensorFlow_outputs(current_path, data_dir, dir_item,split_path = True):
    if not str.endswith(dir_item,'/'):
        dir_item = "%s/"%(dir_item)
    if not split_path:
        directory = dir_item
    else:
        directory = current_path+data_dir+dir_item
    
    TF_DAT={}
    for root, dirs, files in os.walk(directory): 
      # Look inside folder
      matching = [fl for fl in files if fl.endswith('.npz') ]
      if len(matching) == 0:
        continue
      fname=matching[0]
      fullpath = directory+fname
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

def TN_TF_Rsquare(TN, TF):
  """
  With the expectation that TN and TF would generate the same prediction. 
  Thus y = x where x is the neural response from HSM model 
  and y is the neural response from the Tensorflow implementation
  Return the calculate the R^2
  """
  y = TF
  x = TN
  f = x
  err =  y-f
  ymean = np.mean(y)
  SStot = np.sum(np.power(y-ymean,2)) #total sum of squares
  SSreg = np.sum(np.power(f-ymean,2))
  SSres = np.sum(np.power(err,2)) #Residual sum of square
  return 1-np.true_divide(SSres, SStot)

def plot_TN_TF_scatter_linear(TN, TF, titletxt = '',xlbl='Antolik''s implementation with Theano',ylbl="Re-implementation with Tensorflow",RETURN = False):
  line=np.ceil(max(TN.max(),TF.max()))
  fig_handle =plt.figure()
  plt.scatter(TN,TF,c='b',marker='.')
  plt.plot(np.arange(line+1),np.arange(line+1),'-k')
  plt.text(line-1, line-1, 'y = x',
         rotation=45,
         horizontalalignment='center',
         verticalalignment='top',
         multialignment='center')
  plt.title(titletxt)
  plt.xlabel(xlbl)
  plt.ylabel(ylbl)
  if RETURN:
    return fig_handle
  else:
    plt.show()

def computeCorr_flat(response1,response2):
    """
    Compute correlation between two predicted activity flatten version
    """
    if len(response1.shape) >1:
      response1 = response1.flatten()
    if len(response2.shape) >1:
      response2 = response2.flatten()
    if np.all(response1[:]==0) & np.all(response2[:]==0):
        corr=1.
    elif not(np.all(response1[:]==0) | np.all(response2[:]==0)):
        # /!\ To prevent errors due to very low values during computation of correlation
        if abs(response1[:]).max()<1:
            response1[:]=response1[:]/abs(response1[:]).max() 
        if abs(response2[:]).max()<1:
            response2[:]=response2[:]/abs(response2[:]).max()    
        corr=pearsonr(response1,response2)[0]
            
    return corr   
    
# ############# Setting ###############
SAVEFIG=False
dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
Code='SciPy_SEEDnumpy'
hsm = build_hsm_for_Theano()
# ############## Specified Folder ##########################

DATA_LOC = '/media/data_cifs/pachaya/FYPsim/HSMmodel/TFtrainingSummary/'
data_dir =  os.path.join(DATA_LOC,'SciPy_SEEDnumpy')
FIGURES_LOC = os.path.join(os.getcwd(), 'Figures')
figures_dir = os.path.join(FIGURES_LOC,"compare_seeds_%s"%(dt_stamp))

TN_filename = lambda seed,trial:"HSMout_theano_SciPytestSeed_Rg1_MaxIter100000_seed%g-%g.npy"%(seed,trial)
TF_filename = lambda seed,trial:"AntolikRegion1_SciPy_jac_npSeed_MaxIter100000_itr1_SEED%g_trial%g_"%(seed,trial)
TN_list_trial = lambda seed:"HSMout_theano_SciPytestSeed_Rg1_MaxIter100000_seed%g-*"%(seed)
TF_list_trial = lambda seed:"AntolikRegion1_SciPy_jac_npSeed_MaxIter100000_itr1_SEED%g_trial*"%(seed)

"""
this_dir = os.getcwd()+'TFtrainingSummary/SciPy_SEEDnumpy/'

all_TF = glob.glob(os.path.join(data_dir, TF_list_trial(13)))
for f in all_TF:
    print get_param_from_fname(f,'trial')

this_dir = os.getcwd()+'/TFtrainingSummary/SciPy_SEEDnumpy/'
all_TN = glob.glob(os.path.join(this_dir, TN_list_trial(13)))
for f in all_TN:
    print f
    print get_param_from_fname(f,'seed')

ccv_mnt = '/mnt/rdata/data/psailamu/HSMmodel/TFtrainingSummary/SciPy_SEEDnumpy/'
all_TN = glob.glob(os.path.join(ccv_mnt, TN_list_trial(13)))
for f in all_TN:
    print f
    print get_param_from_fname(f,'seed')
"""
# #############  Download Data Set ############## 
download_time = time.time() #print "Download complete: Time %s" %(time.time() - download_time)
runpath=os.getcwd()
NUM_REGIONS =1
train_input ={}; train_set={}; raw_vld_set={}; vldinput_set={}; vld_set={}

for i in range(NUM_REGIONS):
    id=str(i+1)
    train_input[id]=np.load(os.path.join(runpath,'Data/region'+id+'/training_inputs.npy'))
    train_set[id]=np.load(os.path.join(runpath,'Data/region'+id+'/training_set.npy'))
    raw_vld_set[id]=np.load(os.path.join(runpath,'Data/region'+id+'/raw_validation_set.npy'))
    vldinput_set[id]=np.load(os.path.join(runpath,'Data/region'+id+'/validation_inputs.npy'))
    vld_set[id]=np.load(os.path.join(runpath,'Data/region'+id+'/validation_set.npy'))

print "Download Data set complete: Time %s" %(time.time() - download_time)

rg1_train_input = train_input['1']
rg1_train_set = train_set['1']
rg1_vldinput_set = vldinput_set['1']
rg1_vld_set = vld_set['1']

########################################################################
# Check same seed with different trials 
########################################################################

seed_list = np.arange(50) +1
curr_trial = 0
TN_across_seeds ={}
TF_across_seeds ={}
Ks_across_seeds ={}
lgn=9; hlsr=0.2
hsm = HSM(rg1_train_input,rg1_train_set) 
hsm.num_lgn = lgn 
hsm.hlsr = hlsr      
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
   
"""
# Check files
for ss in seed_list:
    #Theano
    fname = TN_filename(ss,curr_trial) 
    this_item = fname[:-4] #remove.npy
    check_file = glob.glob(os.path.join(data_dir,fname))
    if check_file is None:
        print "Error: File not found\t Theano seed = %g"%(ss)
        print this_item
    TF_all_folders = glob.glob(os.path.join(data_dir, TF_filename(ss,curr_trial) +'*'))
    for this_item in TF_all_folders: 
        tmpdat=load_TensorFlow_outputs('', data_dir, this_item,split_path=False)
        if tmpdat is not None:
            break
    if tmpdat is None:
        print "Error: File not found\t Tensorflow seed = %g"%(ss)
        print TF_all_folders
"""
    
    
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

R2_training=[]
R2_VLD =[]
onebig_TN_TR=np.zeros([1800,103,50])
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

allseeds_TN_TR=np.zeros([50,1800*103])
allseeds_TF_TR=np.zeros([50,1800*103])
allseeds_TN_VLD=np.zeros([50,50*103])
allseeds_TF_VLD=np.zeros([50,50*103])

for ss in seed_list:
    id = str(ss)
    allseeds_TN_TR[ss-1,:]= Theano_TR_pred_response[id].flatten()
    allseeds_TF_TR[ss-1,:]=TF_TR_pred_response[id].flatten()
    allseeds_TN_VLD[ss-1,:]=Theano_VLD_pred_response[id].flatten()
    allseeds_TF_VLD[ss-1,:]=TF_VLD_pred_response[id].flatten()
    
    
#scipy.stats.ttest_rel(a, b, axis=0, nan_policy='propagat
#Want to test that the variation for one neuron different from other neuron
#ttest_rel(response1,response2) --- 103 = stats per neuron 
#might have to show the confident interval 
#Like It's might be significantly different but the difference is within 95% CI   
#paired TTest
#TRttest = ttest_rel(onebig_TN_TR.T, onebig_TF_TR.T) #check result from 50 seeds (per neuron, per image)
TRttest = ttest_rel(allseeds_TN_TR, allseeds_TF_TR) #check result from 50 seeds (per neuron, per image)
import numpy as np, statsmodels.stats.api as sms
cm = sms.CompareMeans(sms.DescrStatsW(allseeds_TN_VLD[:,0]), sms.DescrStatsW(allseeds_TF_VLD[:,0]))
print cm.tconfint_diff(usevar='unequal') 


#one samole t-test --- mean diff from 0
diffall = allseeds_TN_VLD.flatten() - allseeds_TF_VLD.flatten()
from scipy import stats
stats.ttest_1samp(diffall,popmean=0.0)
#Ttest_1sampResult(statistic=2.0282379872276359, pvalue=0.042536991866441727)

dd=sms.DescrStatsW(diffall)
sms.DescrStatsW.ttest_mean(dd,value=0, alternative='two-sided')
#http://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.CompareMeans.html

if(False):
    R2_all_regions = {}
    #for ss in seed_list:
    #ss = np.argmax(R2_training)
    ss = np.argmin(R2_training)
    id = str(ss)
    setname ='min TRAINING'
    response1 =Theano_TR_pred_response[id]
    response2 =TF_TR_pred_response[id]
    r_sqr = TN_TF_Rsquare(response1, response2)
    titletxt = "Seed=%g, Trial#%s: Predicted responses from %s set\nR^2 = %f"%(curr_seed, id,setname,r_sqr)
    xlbl='Antolik''s implementation with Theano'
    ylbl="Re-implementation with Tensorflow"

    fig = plot_TN_TF_scatter_linear(response1, response2, titletxt=titletxt,xlbl=xlbl,ylbl=ylbl, RETURN=True)
    R2_all_regions[id] = [TN_TF_Rsquare(this_TN, this_TF) for this_TN, this_TF in zip(response1.T,response2.T)]
    # plt.figure()
    # plt.hist(R2_all_regions[id],50)
    # plt.title(titletxt)

    #ss = np.argmax(R2_VLD)
    ss = np.argmin(R2_VLD)
    id =str(ss)
    setname ='min VALIDATION'
    response1 =Theano_VLD_pred_response[id]
    response2 =TF_VLD_pred_response[id]
    r_sqr = TN_TF_Rsquare(response1, response2)
    titletxt = "Seed=%g, Trial#%s: Predicted responses from %s set\nR^2 = %f"%(curr_seed, id,setname,r_sqr)
    xlbl='Antolik''s implementation with Theano'
    ylbl="Re-implementation with Tensorflow"
    fig =plot_TN_TF_scatter_linear(response1, response2, titletxt=titletxt,xlbl=xlbl,ylbl=ylbl,RETURN=True)
    R2_all_regions[id] = [TN_TF_Rsquare(this_TN, this_TF) for this_TN, this_TF in zip(response1.T,response2.T)]
    # plt.figure()
    # plt.hist(R2_all_regions[id],50)
    # plt.title(titletxt)
    plt.show()
     
    from scipy import stats
    [statistic, pvalue] = stats.ttest_rel(response1, response2)


########################################################################
# Compute correlation coefficient
########################################################################
# 1. corr ---- mea
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
    
def combine_corrs(corrs):
    return np.concatenate((corrs['1'], corrs['2'],corrs['3']),axis=0)

def combine_responses(response):
    return np.concatenate((response['1'], response['2'],response['3']),axis=1) 

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
  plt.text(line_len-1, line_len-1, 'y = x',
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

# ############# Setting ###############
SAVEFIG=True
SEED=13
FIG_HANDLES=[]
FIG_NAMES=[]

# ############## Specified Folder ##########################
Code='TN_TF_comparison_new'
#PATH = '/media/data_cifs/pachaya/SourceCode'

#TFtrainingSummary/SciPy_SEEDnumpy/AntolikRegion3_SciPy_jac_npSeed_MaxIter100000_itr2_SEED13_2017_07_16_00_02_57/TRdat_trainedHSM_region3_trial13.npz

HOST, PATH =get_host_path(HOST=True,PATH=True)

SUMMARY_DIR = 'TFtrainingSummary/SciPy_SEEDnumpy/'

current_path = PATH
data_dir = os.path.join(  current_path, "TFtrainingSummary/SciPy_SEEDnumpy/")

all_dirs = os.listdir(data_dir)

# Save Figure
if SAVEFIG :
    date=str(datetime.now())
    date = date[:10]
    if not os.path.isdir(current_path+'Figures/'+date+'_'+Code+'/') :
      os.mkdir(current_path+'Figures/'+date+'_'+Code+'/')
    Fig_fold=current_path+'Figures/'+date+'_'+Code+'/'
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
#1 seed =13 , all reguons
#2 region =3, seed = 0, 13,13-2
TN_checkSeed ={}; TF_checkSeed = {}
seed_list=['0','13','13-2']
for ss in seed_list:
    TN_checkSeed[ss]=None
    TF_checkSeed[ss]=None
Ks_seeds ={}

Theano_outputs={}; TensorFlow_outputs={}
for i in range(NUM_REGIONS):
    Theano_outputs[str(i+1)]=None
    TensorFlow_outputs[str(i+1)]=None
 
hsm = build_hsm_for_Theano()
Ks={}
for dir_item in all_dirs:
    if str.startswith(dir_item,'HSMout_theano_SciPytestSeed'): #  Theano # HSMout_theano_SciPytestSeed_Rg1_MaxIter100000_seed13
      #remove.npy
      this_item = dir_item[:-4]
      rg_id=get_param_from_fname(this_item, 'Rg');
      seed_id=get_param_from_fname(this_item, 'seed')      
      tmpitem = np.load(data_dir+dir_item)
      if seed_id == '13':
        Theano_outputs[rg_id]=tmpitem.item()
        Theano_outputs[rg_id]['hsm']=hsm[rg_id]
        Ks[rg_id]=Theano_outputs[rg_id]['x']
      if rg_id=='3':
        TN_checkSeed[seed_id]=tmpitem.item()
        TN_checkSeed[seed_id]['hsm']=hsm[rg_id]
        Ks_seeds[seed_id]=TN_checkSeed[seed_id]['x']
      
    elif str.startswith(dir_item,'AntolikRegion'): # Tensorflow
      rg_id=get_param_from_fname(dir_item, 'AntolikRegion')
      seed_id=get_param_from_fname(dir_item, 'SEED')
      tmpdat=load_TensorFlow_outputs('', data_dir, dir_item)
      assert tmpdat is not None 
      if seed_id=='13':
        TensorFlow_outputs[rg_id]=tmpdat
      if rg_id =='3':
        TF_checkSeed[seed_id] = tmpdat
    else:
      continue


Theano_TR_pred_response= {str(id+1):None for id in range(NUM_REGIONS)}
Theano_VLD_pred_response={str(id+1):None for id in range(NUM_REGIONS)}
TF_TR_pred_response={str(id+1):None for id in range(NUM_REGIONS)}
TF_VLD_pred_response={str(id+1):None for id in range(NUM_REGIONS)}

for i in range(NUM_REGIONS):
    id=str(i+1)
    #Theano
    Theano_TR_pred_response[id] = HSM.response(hsm[id],train_input[id],Ks[id]) # predicted response after train
    Theano_VLD_pred_response[id] = HSM.response(hsm[id],vldinput_set[id],Ks[id]) #predicted response for validation set
    #TensorFlow
    TF_TR_pred_response[id] = TensorFlow_outputs[id]['TR_1st_pred_response'] # predicted response after train
    TF_VLD_pred_response[id] = TensorFlow_outputs[id]['VLD_1st_ypredict'] #predicted response for validation set
    

TN_TR_pred_checkseed={ss:None for ss in seed_list}
TN_VLD_pred_checkseed={ss:None for ss in seed_list}
TF_TR_pred_checkseed={ss:None for ss in seed_list}
TF_VLD_pred_checkseed={ss:None for ss in seed_list}
id ='3'
for ss in seed_list:
    #Theano
    TN_TR_pred_checkseed[ss] = HSM.response(hsm[id],train_input[id],Ks_seeds[ss]) # predicted response after train
    TN_VLD_pred_checkseed[ss] = HSM.response(hsm[id],vldinput_set[id],Ks_seeds[ss]) #predicted response for validation set
    #TensorFlow
    TF_TR_pred_checkseed[ss] = TF_checkSeed[ss]['TR_1st_pred_response'] # predicted response after train
    TF_VLD_pred_checkseed[ss] = TF_checkSeed[ss]['VLD_1st_ypredict'] #predicted response for validation set
    
############################################################################################################################
# #############  Comparison ############## 
#Comparison details


#Between-model comparison  TN-TF
# seed =13 , all regions 
# --- R^2 between TN&TF scatter all response
# --- R^2 between TN & TF (per cell) distribution 
R2_all_regions = {}
for i in range(NUM_REGIONS):
  rg = str(i+1)
  setname ='training'
  response1 =Theano_TR_pred_response[rg]
  response2 =TF_TR_pred_response[rg]
  r_sqr = TN_TF_Rsquare(response1, response2)
  titletxt = "Region#%s: Predicted responses from %s set\nR^2 = %f"%(rg,setname,r_sqr)
  xlbl='Antolik''s implementation with Theano'
  ylbl="Re-implementation with Tensorflow"
  plot_TN_TF_scatter_linear(response1, response2, titletxt=titletxt,xlbl=xlbl,ylbl=ylbl)
  R2_all_regions[rg] = [TN_TF_Rsquare(this_TN, this_TF) for this_TN, this_TF in zip(response1.T,response2.T)]
  plt.hist(R2_all_regions[rg],50)
  plt.show()

# region = 3, seed = 0, 13, 13-2
all_cells = [TN_TF_Rsquare(this_TN, this_TF) for this_TN, this_TF in zip(TN.T,TF.T)]
fig=compare_corr_all_regions(Theano_VLD_pred_response,TF_VLD_pred_response, TNTF_vld_corr, stats_param='max', titletxt='Theano&TensorFlow :: Validation Set', RETURN=True)
FIG_HANDLES.append(fig); FIG_NAMES.append('Z0_TNTF_VLD_fig_max')
if(SAVEFIG):
    for fignum in range(len(FIG_HANDLES)):
        fhandle =FIG_HANDLES[fignum]
        fname=FIG_NAMES[fignum]
        fhandle.savefig(Fig_fold+fname+'.png')

#Within-model comparison 
# seed =13 , all regions
response1 =TF_TR_pred_response[Region]
response2 =Theano_TR_pred_response
description = 
# region = 3, seed = 0, 13, 13-2
response1 =''
response2 = ''
description =''


TF=TF_TR_pred_response['3']
TN=Theano_TR_pred_response['3']
line_len = max(np.ceil(TN.max()),np.ceil(TF.max()))

#All cells
all_TN = np.reshape(TN,[-1])
all_TF = np.reshape(TF,[-1])
r_sqr = TN_TF_Rsquare(all_TN, all_TF)
plt.scatter(all_TN,all_TF)
plt.plot(np.arange(line_len+1),np.arange(line_len+1),'k')
plt.text(line_len-1, line_len-1, 'y = x',
         rotation=45,
         horizontalalignment='center',
         verticalalignment='top',
         multialignment='center')
plt.title("Predicted Responses from training set when seed = %g\nR^2 = %f"%(SEED,r_sqr))
plt.xlabel("Antolik's implementation (Theano)")
plt.ylabel("Re-implementation with Tensorflow")
plt.show()

R2_all={}
for s1 in seed_list:
  R2_all[s1] ={}
  for s2 in seed_list:
    setname ='validation'
    response1 =TN_VLD_pred_checkseed[s1]
    response2 =TF_VLD_pred_checkseed[s2]
    r_sqr = TN_TF_Rsquare(response1, response2)
    titletxt = "Predicted responses from %s set\nR^2 = %f"%(setname,r_sqr)
    xlbl="Theano Seed:%s"%s1
    ylbl="Tensorflow Seed:%s"%s2
    fhandle=plot_TN_TF_scatter_linear(response1, response2, titletxt=titletxt,xlbl=xlbl,ylbl=ylbl, RETURN=True)
    R2_all[s1][s2] = [TN_TF_Rsquare(this_TN, this_TF) for this_TN, this_TF in zip(response1.T,response2.T)]
    fname = "Check_seed_VLD_TN%s_TF%s"%(s1,s2)
    fhandle.savefig(Fig_fold+fname+'.png')
    plt.show()

corr_all={}
for s1 in seed_list:
  corr_all[s1] ={}
  for s2 in seed_list:
    setname ='validation'
    response1 =TN_VLD_pred_checkseed[s1]
    response2 =TF_VLD_pred_checkseed[s2]
    corr = computeCorr_flat(response1,response2)
    print "TN:%s TF:%s corr = %f"%(s1,s2,corr)
    corr_all[s1][s2] = corr


corr_TN={}
for s1 in seed_list:
  corr_TN[s1] ={}
  for s2 in seed_list:
    setname ='validation'
    response1 =TN_VLD_pred_checkseed[s1]
    response2 =TN_VLD_pred_checkseed[s2]
    corr = computeCorr_flat(response1,response2)
    print "TN:%s TN:%s corr = %f"%(s1,s2,corr)
    corr_TN[s1][s2] = corr
corr_TN={}
for s1 in seed_list:
  corr_TN[s1] ={}
  for s2 in seed_list:
    setname ='validation'
    response1 =TN_VLD_pred_checkseed[s1]
    response2 =TN_VLD_pred_checkseed[s2]
    corr = computeCorr(response1,response2)
    print "TN:%s TN:%s corr = %f"%(s1,s2,corr.mean())
    corr_TN[s1][s2] = corr.mean()

corr_TF={}
for s1 in seed_list:
  corr_TF[s1] ={}
  for s2 in seed_list:
    setname ='validation'
    response1 =TF_VLD_pred_checkseed[s1]
    response2 =TF_VLD_pred_checkseed[s2]
    corr = computeCorr_flat(response1,response2)
    print "TF:%s TF:%s corr = %f"%(s1,s2,corr)
    corr_TF[s1][s2] = corr


####### DELETE
    response1 =TN_VLD_pred_checkseed['13']
    #response2 =TF_VLD_pred_checkseed['13']
    response2=response1.copy()
    r_sqr = TN_TF_Rsquare(response1, response2)
    titletxt = "Predicted responses from %s set\nR^2 = %f"%(setname,r_sqr)
    xlbl="Theano Seed:%s"%'13'
    ylbl="Tensorflow Seed:%s"%'13'
    fhandle=plot_TN_TF_scatter_linear(response1, response2, titletxt=titletxt,xlbl=xlbl,ylbl=ylbl, RETURN=False)
    plt.show()

plot_seeds_withR2(TF_TR_pred_checkseed['13'],TF_TR_pred_checkseed['13-2'],set_name='training',region='3',seeds =('13','13-2'), implementation = 'Re-implementation with Tensorflow')

plot_seeds_withR2(TN_TR_pred_checkseed['13'],TN_TR_pred_checkseed['13-2'],set_name='training',region='3',seeds =('13','13-2'), implementation = 'Antolik''s implementation')


plot_seeds_withR2(TF_VLD_pred_checkseed['13'],TF_VLD_pred_checkseed['13-2'],set_name='training',region='3',seeds =('13','13-2'), implementation = 'Re-implementation with Tensorflow')

plot_seeds_withR2(TN_VLD_pred_checkseed['13'],TN_VLD_pred_checkseed['13-2'],set_name='training',region='3',seeds =('13','13-2'), implementation = 'Antolik''s implementation')



plot_seeds_withR2(TN_VLD_pred_checkseed['13'],TN_VLD_pred_checkseed['13-2'],seeds =('0','13'), implementation ='')


cell =2
this_TN=TN[:,cell]
this_TF=TF[:,cell]
line=np.ceil(max(this_TN.max(),this_TF.max()))

plt.scatter(this_TN,this_TF)
plt.plot(np.arange(line+1),np.arange(line+1),'k')
plt.show()

r_sqr = TN_TF_Rsquare(this_TN, this_TF)

plot_TN_TF_scatter_linear(this_TN, this_TF)
plot_TN_TF_withR2(this_TN,this_TF,set_name='validation',SEED=13)

all_cells = [TN_TF_Rsquare(this_TN, this_TF) for this_TN, this_TF in zip(TN.T,TF.T)]



from visualization import *
corr = computeCorr(TN, TF)
vld_corr = computeCorr(TN,TF)


from funcs_for_graphs import *
Region_num='3'
report_txt="Region #%s: Training Set\nMean corr = %.4f, best neuron has corr = %.4f, median neuron=%.4f"%(Region_num, corr.mean(), corr.max(), np.median(corr))
plot_act_of_max_min_corr(report_txt, TN,TF,corr, PLOT=True,ZOOM=True)

report_txt="Region #%s: Validation Set\nMean corr = %.4f, best neuron has corr = %.4f, median neuron=%.4f"%(Region_num, vld_corr.mean(), vld_corr.max(), np.median(vld_corr))
plot_act_of_max_min_corr(report_txt, pred_response,vld_set,vld_corr, PLOT=True,ZOOM=False)




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
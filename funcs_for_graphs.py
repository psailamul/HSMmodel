#visualization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from HSM import HSM
from get_host_path import get_host_path
import os
import re
from scipy.stats import pearsonr
import glob
import param

SEED = 13



def computeCorr(pred_act,responses):
    """
    Compute correlation between predicted and recorded activity for each cell
    """
    #import ipdb; ipdb.set_trace()
    num_pres,num_neurons = np.shape(responses)
    corr=np.zeros(num_neurons)
    
    for i in xrange(0,num_neurons):
        if np.all(pred_act[:,i]==0) & np.all(responses[:,i]==0):
            corr[i]=1.
        elif not(np.all(pred_act[:,i]==0) | np.all(responses[:,i]==0)):
            # /!\ To prevent errors due to very low values during computation of correlation
            if abs(pred_act[:,i]).max()<1:
                pred_act[:,i]=pred_act[:,i]/abs(pred_act[:,i]).max()
            if abs(responses[:,i]).max()<1:
                responses[:,i]=responses[:,i]/abs(responses[:,i]).max()    
            corr[i]=pearsonr(np.array(responses)[:,i].flatten(),np.array(pred_act)[:,i].flatten())[0]
            
    return corr


def printCorrelationAnalysis(act,val_act,pred_act,pred_val_act):
    """
    This function simply calculates the correlation between the predicted and 
    and measured responses for the training and validation set and prints them out.
    """
    train_c=computeCorr(pred_act,act)
    val_c=computeCorr(pred_val_act,val_act)
    
    print 'Correlation Coefficients (training/validation): ' + str(np.mean(train_c)) + '/' + str(np.mean(val_c))
    return (train_c,val_c)
    
# functions for data visualization

def hist_of_pred_and_record_response(runcodestr, pred_response, recorded_response, cell_id=0, PLOT=False):
  fig,ax = plt.subplots(figsize=(12,8))
  plt.subplot(121); plt.hist(recorded_response[:,cell_id]); plt.title('Recorded Response');
  plt.subplot(122); plt.hist(pred_response[:,cell_id]); plt.title('Predicted Response');
  plt.suptitle("Code: %s\nDistribution of cell #%g's response"%(runcodestr,cell_id))
  if(PLOT):
   plt.show()
  return fig,ax

def plot_act_of_max_min_corr(runcodestr, yhat,train_set,corr, PLOT=False,ZOOM=False,RETURN=False,TITLE=True):
    if RETURN:
        PLOT=False
    fig_max_z = None
    fig_min_z=None
    imax = np.argmax(corr) # note : actually have to combine neurons in all regions
    fig_max =plt.figure()
    plt.plot(train_set[:,imax],'-ok')
    plt.plot(yhat[:,imax],'--or')
    if TITLE:
        plt.title('Code: %s\nCell#%d has max corr of %f'%(runcodestr,imax+1,np.max(corr)))
    if(PLOT):
      plt.show()
    if(ZOOM):
      fig_max_z =plt.figure()
      plt.plot(train_set[:,imax],'-ok')
      plt.plot(yhat[:,imax],'--or')
      plt.xlim((600,650))
      if TITLE:
        plt.title('Code: %s\nCell#%d has max corr of %f'%(runcodestr,imax+1,np.max(corr)))
      if(PLOT):
        plt.show()
      
    imin = np.argmin(corr) # note : actually have to combine neurons in all regions
    fig_min = plt.figure()
    plt.plot(train_set[:,imin],'-ok')
    plt.plot(yhat[:,imin],'--or')
    if TITLE:
        plt.title('Code: %s\nCell#%d has min corr of %f'%(runcodestr,imin+1,np.min(corr)))
    if(PLOT):
      plt.show()
    if(ZOOM):
      fig_min_z =plt.figure()
      plt.plot(train_set[:,imin],'-ok')
      plt.plot(yhat[:,imin],'--or')
      plt.xlim((600,650))
      if TITLE:
        plt.title('Code: %s\nCell#%d has min corr of %f'%(runcodestr,imin+1,np.min(corr)))
      if(PLOT):
        plt.show()
    if RETURN:
        if ZOOM:
            return fig_max, fig_max_z, fig_min, fig_min_z
        else:
            return fig_max, fig_min

def compare_corr_all_regions(pred_response,vld_set, corr_set, stats_param='max',titletxt='', RETURN=False):
    corr1=corr_set['1']; corr2=corr_set['2']; corr3=corr_set['3']; 
    
    if stats_param.lower() == 'max' :
        idx1=np.argmax(corr1); idx2=np.argmax(corr2); idx3=np.argmax(corr3)
        stat1 = np.max(corr1); stat2 = np.max(corr2); stat3 = np.max(corr3)
    elif stats_param.lower() == 'min' :
        idx1=np.argmin(corr1); idx2=np.argmin(corr2); idx3=np.argmin(corr3)
        stat1 = np.min(corr1); stat2 = np.min(corr2); stat3 = np.min(corr3); 
    elif stats_param.lower() == 'median' :
        N1=len(corr1); N2=len(corr2); N3=len(corr3);
        idx1 = N1/2-1 if N1%2==0 else (N1-1)/2
        idx2 = N2/2-1 if N2%2==0 else (N2-1)/2
        idx3 = N3/2-1 if N3%2==0 else (N3-1)/2
        
        stat1=np.sort(corr1)[idx1]
        stat2=np.sort(corr2)[idx2]
        stat3=np.sort(corr3)[idx3]
    else:
        print "Parameter not Found "
    combine_corr = np.concatenate((corr1, corr2,corr3),axis=0)

    fig, ax = plt.subplots()
    plt.subplot(311)
    plt.plot(vld_set['1'][:,idx1],'-ok')
    plt.plot(pred_response['1'][:, idx1],'--or')
    plt.title('Cell#%g is the %s  neuron in region 1, R = %.5f, mean neuron has R = %.5f'%(idx1+1 ,stats_param,stat1, np.mean(corr1)))

    plt.subplot(312)
    plt.plot(vld_set['2'][:,idx2],'-ok')
    plt.plot(pred_response['2'][:, idx2],'--or')
    plt.title('Cell#%g is the %s neuron in region 2, R = %.5f, mean neuron has R = %.5f'%(idx2+1 ,stats_param,stat2, np.mean(corr2)))
    
    plt.subplot(313)
    plt.plot(vld_set['3'][:,idx3],'-ok')
    plt.plot(pred_response['3'][:, idx3],'--or')
    plt.title("Cell#%g is the %s neuron in region 3, R = %.5f, mean neuron has R = %.5f"%(idx3+1 ,stats_param,stat3, np.mean(corr3)))
    
    report_txt="Overall mean corr = %.4f, best neuron has corr = %.4f, median neuron=%.4f"%(combine_corr.mean(), 
        combine_corr.max(), np.median(combine_corr))
            
    if titletxt=='':
        plt.suptitle(report_txt)
    else:
        plt.suptitle("%s\n%s"%(titletxt,report_txt))
    plt.show()
    
    if RETURN :
        return fig
        
def cdf_allregions( CorrData, NUM_REGIONS=3, DType='',fileloc='',filename='CDF' , C_CODE=False, SHOW=True, RETURN=False, SAVE=False):
    fig=plt.figure()
    Fs={}; Xs={}
    if(C_CODE):
        Ccode=('k','b','#F97306')
    for id in range(NUM_REGIONS):
      rg=str(id+1)
      corr = CorrData[rg]
      N=len(corr)
      Xs[rg]= np.sort(corr)
      Fs[rg] = np.array(range(N))/float(N)*100.0
      if(C_CODE):
        plt.step(Xs[rg], Fs[rg],color= Ccode[id], label='Region'+rg)
      else:
        plt.step(Xs[rg], Fs[rg], label='Region'+rg)
    
    plt.legend(loc='upper left')
    plt.title(DType+' CDF')
    plt.ylabel('% of neurons')
    plt.xlabel('Correlation coefficient')
    if SAVE:
        plt.savefig(os.path.join(fileloc,filename+'.svg'))
        plt.savefig(os.path.join(fileloc,filename+'.png'))
    if(SHOW):
        plt.show()
    if(RETURN):
        return fig, Xs, Fs

def plot_corr_response_scatter(pred_response, vld_set, corr_set, stats_param='max',titletxt='', RETURN=False, datalabel1='Measured Response', datalabel2='Predicted Response'):
    
    if stats_param.lower() == 'max' :
        idx1=np.argmax(corr_set);
        stat1 = np.max(corr_set); 
    elif stats_param.lower() == 'min' :
        idx1=np.argmin(corr_set);
        stat1 = np.min(corr_set); 
    elif stats_param.lower() == 'median' or stats_param.lower() == 'med':
        N1=len(corr_set);
        idx1 = N1/2-1 if N1%2==0 else (N1-1)/2
        stat1=np.sort(corr_set)[idx1]
    else:
        print "Parameter not Found "

    fig, ax = plt.subplots(figsize=[16,4])
    plt.subplot(1,2,1)
    plt.plot(vld_set[:,idx1],'-ok',label=datalabel1)
    plt.plot(pred_response[:, idx1],'--or',label=datalabel2)
    plt.ylabel('Response')
    plt.xlabel('Image #')
    plt.title("Cell#%g is the %s  neuron, R = %.5f, mean neuron has R = %.5f"%(idx1+1 ,stats_param,stat1, np.mean(corr_set)))
    plt.legend(loc=0)
    
    plt.subplot(1,2,2)
    plt.scatter(vld_set[:,idx1], pred_response[:,idx1])
    N=np.ceil(np.max([np.max(pred_response[:,idx1]),np.max(vld_set[:,idx1])]))
    plt.plot(np.arange(N),np.arange(N),'--c')
    plt.xlim([0,N]); plt.ylim([0,N])
    plt.ylabel(datalabel2)
    plt.xlabel(datalabel1)
    plt.title('Scatter plot of measured response and predicted response of cell#%g'%(idx1+1))  

    if titletxt is not '':
        plt.suptitle(titletxt)
    plt.show()
    
    if RETURN :
        return fig

def plot_fig4_response_scatter( model_activity, 
                                cell_true_activity, 
                                corr_set, 
                                stats_param='max',
                                fileloc='',
                                filename='fig4',
                                runcodestr='',
                                datalabel1='Recorded neural response', 
                                datalabel2='Predicted response from model',
                                RETURN=False, 
                                SAVE=False
                                ):
    
    if stats_param.lower() == 'max' :
        idx1=np.argmax(corr_set);
        stat1 = np.max(corr_set);
        nr_text ='best'
    elif stats_param.lower() == 'min' :
        idx1=np.argmin(corr_set);
        stat1 = np.min(corr_set);
        nr_text ='worst'
    elif stats_param.lower() == 'median' or stats_param.lower() == 'med':
        N1=len(corr_set);
        idx1 = N1/2-1 if N1%2==0 else (N1-1)/2
        stat1=np.sort(corr_set)[idx1]
        nr_text ='median'
    else:
        print "Parameter not Found "
    
    #Compare response
    fig, ax = plt.subplots(figsize=[16,5])
    plt.subplot(1,2,1)
    plt.plot(cell_true_activity[:,idx],'-ok',label=datalabel1)
    plt.plot(model_activity[:,idx], '--ok', markerfacecolor='white',label=datalabel2)
    ylim_up = np.ceil(np.max([np.max(cell_true_activity[:,idx]),np.max(model_activity[:,idx])]))
    ylim_low = np.floor(np.min([np.min(cell_true_activity[:,idx]),np.min(model_activity[:,idx])]))
    plt.ylim([ylim_low,ylim_up])
    plt.xlabel('Image #')
    plt.ylabel('Response')
    plt.legend(loc=0)

    #scatter plot
    plt.subplot(1,2,2)
    plt.scatter(cell_true_activity[:,idx1], model_activity[:,idx1], facecolors='none', edgecolors='k')
    N=np.ceil(np.max([np.max(model_activity[:,idx1]),np.max(cell_true_activity[:,idx1])]))
    plt.plot(np.arange(N),np.arange(N),'--',color=(0.6,0.6,0.6),label='reference line(y = x)')
    plt.xlim([0,N]); plt.ylim([0,N])
    plt.ylabel(datalabel2)
    plt.xlabel(datalabel1)
    plt.legend(loc=0)
    plt.suptitle('%s\nResponse per image of the %s neuron, R=%f'%(runcodestr,nr_text,stat1))
    if SAVE:
        plt.savefig(os.path.join(fileloc,filename+'.svg'))
        plt.savefig(os.path.join(fileloc,filename+'.png'))
    plt.show()
    
    if RETURN :
        return fig
        
def plot_seeds_withR2(S1,S2,set_name='validation',region='3',seeds =('13','0'), implementation = 'Antolik''s implementation'):

  line_len = max(np.ceil(S1.max()),np.ceil(S2.max()))

  #All cells
  all_S1 = np.reshape(S1,[-1])
  all_S2 = np.reshape(S2,[-1])
  r_sqr = TN_TF_Rsquare(all_S1, all_S2)
  plt.scatter(all_S1,all_S2)
  plt.plot(np.arange(line_len+1),np.arange(line_len+1),'k')
  plt.text(line_len-1, line_len-1, 'y = x',
           rotation=45,
           horizontalalignment='center',
           verticalalignment='top',
           multialignment='center')
  plt.title("Predicted Responses from %s set in region = %s\nR^2 = %f"%(set_name,region,r_sqr))
  plt.xlabel("%s seed: %s"%(implementation,seeds[0]))
  plt.ylabel("%s seed: %s"%(implementation,seeds[1]))
  plt.show()

def plot_TN_TF_withR2(TN,TF,set_name='validation',SEED=SEED):
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
  plt.title("Predicted Responses from %s set when seed = %g\nR^2 = %f"%(set_name,SEED,r_sqr))
  plt.xlabel("Antolik's implementation (Theano)")
  plt.ylabel("Re-implementation with Tensorflow")
  plt.show()

def plot_TN_TF_scatter_linear(TN, TF, titletxt = '',xlbl='Antolik''s implementation with Theano',ylbl="Re-implementation with Tensorflow"):
  line=np.ceil(max(TN.max(),TF.max()))
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
  plt.show()
  
#plot previdted response TN vs TF and R^2 
#plot previdted response vs measured for all types and R^2 

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
    
    
def get_param_from_fname(fname, keyword):
    cuts = re.split('_',fname)
    for prm in cuts:
        if str.startswith(prm,keyword):
            return prm[len(keyword):]
    else:
        print "WARNING:: KEYWORD NOT FOUND"
        return None

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
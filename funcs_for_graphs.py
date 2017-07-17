#visualization

import numpy as np
import matplotlib.pyplot as plt
from visualization import *


# functions for data visualization

def hist_of_pred_and_record_response(runcodestr, pred_response, recorded_response, cell_id=0, PLOT=False):
  fig,ax = plt.subplots(figsize=(12,8))
  plt.subplot(121); plt.hist(recorded_response[:,cell_id]); plt.title('Recorded Response');
  plt.subplot(122); plt.hist(pred_response[:,cell_id]); plt.title('Predicted Response');
  plt.suptitle("Code: %s\nDistribution of cell #%g's response"%(runcodestr,cell_id))
  if(PLOT):
   plt.show()
  return fig,ax

def plot_act_of_max_min_corr(runcodestr, yhat,train_set,corr, PLOT=False,ZOOM=False):
    
    fig_max_z = None
    fig_min_z=None
    imax = np.argmax(corr) # note : actually have to combine neurons in all regions
    fig_max =plt.figure()
    plt.plot(train_set[:,imax],'-ok')
    plt.plot(yhat[:,imax],'--or')
    plt.title('Code: %s\nCell#%d has max corr of %f'%(runcodestr,imax+1,np.max(corr)))
    if(PLOT):
      plt.show()
    if(ZOOM):
      fig_max_z =plt.figure()
      plt.plot(train_set[:,imax],'-ok')
      plt.plot(yhat[:,imax],'--or')
      plt.xlim((600,650))
      plt.title('Code: %s\nCell#%d has max corr of %f'%(runcodestr,imax+1,np.max(corr)))
      if(PLOT):
        plt.show()
      
    imin = np.argmin(corr) # note : actually have to combine neurons in all regions
    fig_min = plt.figure()
    plt.plot(train_set[:,imin],'-ok')
    plt.plot(yhat[:,imin],'--or')
    plt.title('Code: %s\nCell#%d has min corr of %f'%(runcodestr,imin+1,np.min(corr)))
    if(PLOT):
      plt.show()
    if(ZOOM):
      fig_min_z =plt.figure()
      plt.plot(train_set[:,imin],'-ok')
      plt.plot(yhat[:,imin],'--or')
      plt.xlim((600,650))
      plt.title('Code: %s\nCell#%d has min corr of %f'%(runcodestr,imin+1,np.min(corr)))
      if(PLOT):
        plt.show()

def  compare_corr_all_regions(pred_response,vld_set, corr_set, stats_param='max',titletxt='', RETURN=False):
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
        
def cdf_allregions( CorrData, NUM_REGIONS=3, DType='', C_CODE=False, SHOW=True, RETURN=False):
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
    if(SHOW):
        plt.show()
    if(RETURN):
        return fig, Xs, Fs

def  plot_corr_response_scatter(pred_response, vld_set, corr_set, stats_param='max',titletxt='', RETURN=False, datalabel1='Measured Response', datalabel2='Predicted Response'):
    
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

    fig, ax = plt.subplots()
    plt.subplot(2,1,1)
    plt.plot(vld_set[:,idx1],'-ok',label=datalabel1)
    plt.plot(pred_response[:, idx1],'--or',label=datalabel2)
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
    

    if titletxt is not '':
        plt.suptitle(titletxt)
    plt.show()
    
    if RETURN :
        return fig

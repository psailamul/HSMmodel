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

def  compare_corr_all_regions(pred_response,vld_set, corr_set, stats_param='max',titletxt=''):
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
        
        srt1 = np.sort(corr1); stat1=np.sort(corr1)[idx1]
        stat1=corr1[idx1]; stat2=corr2[idx2]; stat3=corr3[idx3]; 
    else:
        print "Parameter not Found "
    combine_corr = np.concatenate((corr1, corr2,corr3),axis=0)
    
    ax1=plt.subplot(3,1,1)
    ax1.plot(vld_set['1'][:,idx1],'-ok')
    ax1.plot(pred_response['1'][:, idx1],'--or')
    ax1.set_title('Cell#%g is the %s  neuron in region 1, R = %.5f, mean neuron has R = %.5f'%(idx1+1 ,stats_param,stat1, np.mean(corr1)))

    ax2=plt.subplot(3,1,2)
    ax2.plot(vld_set['2'][:,idx2],'-ok')
    ax2.plot(pred_response['2'][:, idx2],'--or')
    ax2.set_title('Cell#%g is the %s neuron in region 2, R = %.5f, mean neuron has R = %.5f'%(idx2+1 ,stats_param,stat2, np.mean(corr2)))
    
    ax3=plt.subplot(3,1,3)
    ax3.plot(vld_set['3'][:,idx3],'-ok')
    ax3.plot(pred_response['3'][:, idx3],'--or')
    ax3.set_title("Cell#%g is the %s neuron in region 3, R = %.5f, mean neuron has R = %.5f"%(idx3+1 ,stats_param,stat3, np.mean(corr3)))
    report_txt="Overall mean corr = %.4f, best neuron has corr = %.4f, median neuron=%.4f"%(combine_corr.mean(), 
        combine_corr.max(), np.median(combine_corr))
            
    if titletxt=='':
        plt.suptitle(report_txt)
    else:
        plt.suptitle("%s\n%s"%(titletxt,report_txt))
    plt.show()
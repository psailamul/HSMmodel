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
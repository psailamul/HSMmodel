# load training of all restart then analyzed
import numpy as np
import tensorflow as tf
import param
from tf_HSM import tf_HSM
import os
import matplotlib.pyplot as plt
from datetime import datetime
import re 
from tensorflow.python import debug 
import time
from visualization import *
import sys


def correlate_vectors(yhat_array, y_array):
  corrs = []      
  for yhat, y in zip(np.transpose(yhat_array), np.transpose(y_array)):
    tc = np.corrcoef(yhat, y)[1,0]
    if np.isnan(tc):
      tc=0.0
    #tc = (np.isnan(tc) == False).astype(float) * tc
    corrs += [tc]
  return corrs

def log_likelihood(predictions,targets,epsilon =0.0000000000000000001):
  return tf.reduce_sum(predictions) - tf.reduce_sum(tf.multiply(targets,tf.log(predictions + epsilon)))

def hist_of_pred_and_record_response(pred_response, recorded_response, cell_id=0):
  plt.subplot(121); plt.hist(recorded_response[:,cell_id]); plt.title('Recorded Response');
  plt.subplot(122); plt.hist(pred_response[:,cell_id]); plt.title('Predicted Response');
  plt.suptitle("Distribution of cell #%g's response"%cell_id)
  plt.show()

def plot_act_of_max_min_corr(yhat,train_set,corr):
    plt.figure()
    imax = np.argmax(corr) # note : actually have to combine neurons in all regions

    plt.plot(train_set[:,imax],'-ok')
    plt.plot(yhat[:,imax],'--or')
    plt.title('Cell#%d has max corr of %f'%(imax+1,np.max(corr)))
    plt.show()

    imin = np.argmin(corr) # note : actually have to combine neurons in all regions

    plt.plot(train_set[:,imin],'-ok')
    plt.plot(yhat[:,imin],'--or')
    plt.title('Cell#%d has min corr of %f'%(imin+1,np.min(corr)))
    plt.show()

        # check the training
def plot_training_behav(runcodestr,loss_list,MSE_list,corr_list,activation_summary_lgn,activation_summary_l1,yhat_std,lr=1E-03,iterations=2500,SAVE=True,PLOT=False):
    itr_idx = range(iterations)
    plt.figure()
    fig,ax = plt.subplots(figsize=(16,12))
    plt.subplot(2, 3, 1)
    plt.plot(itr_idx, loss_list, 'k-')
    plt.title('Loss')
    plt.xlabel('iterations')

    plt.subplot(2, 3, 2)
    plt.plot(itr_idx, MSE_list, 'b-')
    plt.title('MSE')
    plt.xlabel('iterations')

    plt.subplot(2, 3, 3)
    plt.plot(itr_idx, corr_list, 'r-')
    plt.title('Mean Correlation')
    plt.xlabel('iterations')

    plt.subplot(2, 3, 4)
    plt.plot(itr_idx, activation_summary_lgn, 'r-')
    plt.title('Mean LGN activation')
    plt.xlabel('iterations')

    plt.subplot(2, 3, 5)
    plt.plot(itr_idx, activation_summary_l1, 'r-')
    plt.title('Mean L1 activation')
    plt.xlabel('iterations')

    plt.subplot(2,3, 6)
    plt.plot(itr_idx, yhat_std, 'r-')
    plt.title('std of predicted response')
    plt.xlabel('iterations')

    plt.suptitle("Code: %s lr=%.5f , itr = %g\n[%s]"%(runcodestr,lr,iterations,str(datetime.now())))
    plt.show()
    return fig, ax
    """
    if(PLOT):
        plt.show()
    if(SAVE):
        fig.savefig('test2png.png')
    """
def get_trials_seednum(fullpath):
  fullpath=fullpath.replace('.','_')
  cutstr = re.split('_',fullpath)
  matching = [s for s in cutstr if "trial" in s]
  assert matching is not None
  trial_num = int(matching[0][5:])

  matching = [s for s in cutstr if "seed" in s]
  assert matching is not None
  seed_num = int(matching[0][4:])
  return trial_num, seed_num

def main():
  dt_stamp = re.split(
    '\.', str(datetime.now()))[0].\
    replace(' ', '_').replace(':', '_').replace('-', '_')
  region_num='1'
  lr=1E-03; iterations=500

  current_path = os.getcwd()+'/'
  #data_dir = os.path.join(  "TFtrainingSummary/Region"+region_num+'/')
  data_dir = os.path.join(  "TFtrainingSummary/500itr/")
  sim_folder = "AntolikRegion1_lr0.00100_itr500_2017_06_27_19_36_30/"
  fname = "TRdat_trained_HSM_3_trial50_seed50.npz" 
  fullpath = data_dir+sim_folder+fname
  npz_dat = np.load(fullpath)


  trial_num, seed_num = get_trials_seednum(fullpath)

  for k in npz_dat:
    exec("{}".format(k)+"= npz_dat.f."+"{}".format(k))
  
  CONFIG=CONFIG.all()
  runcodestr = CONFIG['runcodestr']
  train_set=np.load(current_path+'Data/region' + region_num+'/training_set.npy')
  
  if(CONFIG['NORM_RESPONSE']):
    train_set = train_set/(train_set.max()-train_set.min())


  fig, ax=plot_training_behav(runcodestr = runcodestr,
   loss_list=TR_loss,
   MSE_list=TR_MSE,
   corr_list=TR_corr,
   activation_summary_lgn=TR_mean_LGNact,
   activation_summary_l1=TR_mean_L1act,
   yhat_std=TR_std_pred_response,
   lr=lr, iterations=iterations)

  responses=train_set
  pred_act = TR_last_pred_response;
  #pred_act = TR_1st_pred_response;
  corr = computeCorr(pred_act, train_set)
  corr[np.isnan(corr)]=0.0
  plot_act_of_max_min_corr(pred_act,responses,corr)
  hist_of_pred_and_record_response(pred_act,responses,cell_id=np.argmax(corr))
     
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
  return tf.reduce_sum(predictions) - tf.reduce_sum(tf.mul(targets,tf.log(predictions + epsilon)))

def hist_of_pred_and_record_response(runcodestr, pred_response, recorded_response, cell_id=0, PLOT=False):
  fig,ax = plt.subplots(figsize=(12,8))
  plt.subplot(121); plt.hist(recorded_response[:,cell_id]); plt.title('Recorded Response');
  plt.subplot(122); plt.hist(pred_response[:,cell_id]); plt.title('Predicted Response');
  plt.suptitle("Code: %s\nDistribution of cell #%g's response"%(runcodestr,cell_id))
  if(PLOT):
   plt.show()
  return fig,ax

def plot_act_of_max_min_corr(runcodestr, yhat,train_set,corr, PLOT=False):
    
    imax = np.argmax(corr) # note : actually have to combine neurons in all regions
    fig_max =plt.figure()
    plt.plot(train_set[:,imax],'-ok')
    plt.plot(yhat[:,imax],'--or')
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

    return fig_max, fig_min

        # check the training
def plot_training_behav(runcodestr,loss_list,MSE_list,corr_list,activation_summary_lgn,activation_summary_l1,yhat_std,lr=1E-03,iterations=2500,PLOT=False):
    itr_idx = range(iterations)

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
    
    if(PLOT):
        plt.show()
    return fig, ax

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

SAVEFIG = True
PLOT=True

def main(region_num='1', lr=1E-03, iterations=100000):
    # Read file
  if SAVEFIG :
    date=str(datetime.now())
    date = date[:10]
    if not os.path.isdir('Figures/'+date+'/') :
      os.mkdir('Figures/'+date+'/')
    Fig_fold='Figures/'+date+'/'

  dt_stamp = re.split(
      '\.', str(datetime.now()))[0].\
      replace(' ', '_').replace(':', '_').replace('-', '_')
  
  current_path = os.getcwd()+'/'
  #data_dir = os.path.join(  "TFtrainingSummary/Region"+region_num+'/')
  data_dir = os.path.join(  "TFtrainingSummary/LargeIterations_100k/")

  all_folders = os.listdir(current_path+data_dir)
  for sim_folder in all_folders:
    sim_folder+='/'
    directory = current_path+data_dir+sim_folder
    print("------------------------------------------")
    print(sim_folder)
    print("------------------------------------------")
    run_time = time.time()
    #sim_folder ="AntolikRegion1_lr0.00100_itr2500_2017_06_23_06_39_51/"
    for root, dirs, files in os.walk(directory): 

      # Look inside each folder
      matching = [file for file in files if file.endswith('.npz')]
      if matching is None:
        continue
      fname=matching[0]
      #import ipdb; ipdb.set_trace()
      #fname = "TRdat_trained_HSM_3_trial50_seed50.npz"
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
          
          
      fig_train, ax=plot_training_behav(runcodestr = runcodestr,
           loss_list=TR_loss,
           MSE_list=TR_MSE,
           corr_list=TR_corr,
           activation_summary_lgn=TR_mean_LGNact,
           activation_summary_l1=TR_mean_L1act,
           yhat_std=TR_std_pred_response,
           lr=lr, iterations=iterations, PLOT=PLOT)

      responses = train_set
      pred_act = TR_last_pred_response;
      corr = computeCorr(pred_act, train_set)
      corr[np.isnan(corr)]=0.0
      fig_max,fig_min = plot_act_of_max_min_corr(runcodestr,pred_act,responses,corr,PLOT=PLOT)
      fig_hist, ax = hist_of_pred_and_record_response(runcodestr,pred_act,responses,cell_id=np.argmax(corr),PLOT=PLOT)

      if SAVEFIG:
        sim_code="Region%s_lr%.5f_itr%g_trial%g_seed%g"%(region_num,lr,iterations,trial_num,seed_num)
        fig_train.savefig(Fig_fold+sim_code+"_TrainRec.png")
        fig_max.savefig(Fig_fold+sim_code+"_MaxCorr.png")
        fig_min.savefig(Fig_fold+sim_code+"_MinCorr.png")
        fig_hist.savefig(Fig_fold+sim_code+"_ResponseDist.png")

      plt.close("all")      
      print "Saved: Time %s\n" %(time.time() - run_time)

if __name__ == "__main__":
  region_num='1'
  tt_run_time = time.time()
  main(region_num=region_num)
  print "Finished Region"+region_num+ ": Time %s\n" %(time.time() - tt_run_time)

     
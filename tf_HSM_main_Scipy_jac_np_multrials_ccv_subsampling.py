import numpy as np
import tensorflow as tf
import param
from tf_HSM_upgraded_SciPy_Numpy import tf_HSM
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib.pyplot as plt
from datetime import datetime
import re 
from tensorflow.python import debug 
import time
from visualization import *
import sys
from get_host_path import get_host_path
ScipyOptimizerInterface = tf.contrib.opt.ScipyOptimizerInterface
from numpy.random import seed, rand
    
# CUDA_VISIBLE_DEVICES=2 python tf_HSM_main_MaxFunc_100k_upgraded.py REGION=2 RESTART_TRIAL=0 SEED=13 ITERATIONS=1 LR=1e-3 NUM_LGN=9 HLSR=0.2
# python tf_HSM_main_Scipy_jac_np_multrials.py | tee Seed_LOG/HSM_Tensorflow_region3_seed13.txt
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
    imax = np.argmax(corr) # note : actually have to combine neurons in all regions

    plt.plot(train_set[:,imax],'-ok')
    plt.plot(yhat[:,imax],'--or')
    plt.title('Cell#%d has max corr of %f'%(imax+1,np.max(corr)))
    plt.show()

    imin = np.argmin(corr) # note : actually have to combine neurons in all regions

    plt.plot(train_set[:,imin],'-ok')
    plt.plot(yhat[:,imin],'--or')
    plt.title('Cell#%d has min corr of %f'%(imin+1,np.min(corr)))
    
#########################################################################
# Main script
########################################################################

def main():
    #########################################################################
    # Simulation Config
    ########################################################################

    REGION =1; RESTART_TRIAL=1; SEED =13; ITERATIONS=1; LR = 1e-3; NUM_LGN=9; HLSR=0.2;
    MAXITER = 100000
    if len(sys.argv) > 1:
        for ii in range(1,len(sys.argv)):
            arg = sys.argv[ii]
            print(arg)
            exec(arg) 
    #tf.set_random_seed(SEED)
    print('SEED : %g'%(SEED))
    tt_run_time = time.time()
    runID = RESTART_TRIAL
    dt_stamp = re.split(
            '\.', str(datetime.now()))[0].\
            replace(' ', '_').replace(':', '_').replace('-', '_')
    HOST, PATH = get_host_path(HOST=True, PATH=True)
    region_num = str(REGION)
    num_lgn=NUM_LGN; hlsr=HLSR
    lr = LR
    iterations = ITERATIONS
    NORM_RESPONSE = False
    SAVEdat = True
    VISUALIZE = False
    PLOT_CORR_STATS =False
    runcodestr ="Machine: %s Region: %s max iter: %g Iterations: %g Restart#: %g"%(HOST,region_num, MAXITER, iterations, RESTART_TRIAL)
    #SAVER_SAVE = int(iterations/10.0)
    #SAVER_SAVE = np.max(1,SAVER_SAVE)
    SAVER_SAVE=1
    CONFIG={'region_num':region_num,
    'runcodestr':runcodestr,
    'NORM_RESPONSE':NORM_RESPONSE,
    'SAVEdat':SAVEdat,
    'VISUALIZE':VISUALIZE,
    'PLOT_CORR_STATS':PLOT_CORR_STATS,
    'RESTART_TRIAL':RESTART_TRIAL,
    'SEED':SEED,
    'ITERATIONS':ITERATIONS,
    'LR':LR,
    'NUM_LGN': NUM_LGN,
    'HLSR':HLSR,
    'MAXITER':MAXITER
    }
    
    #########################################################################
    # Directories
    ########################################################################
    Code='subsampling'
    MAIN_LOC= '/users/psailamu/' #CCV 
    #MAIN_LOC = '/mnt/rdata/' #On x8

    DATA_LOC = MAIN_LOC+'data/psailamu/AntolikData'
    OUTPUT_LOC = MAIN_LOC+'data/psailamu/HSMmodel/TFtrainingSummary/subsampling/' 
    SUMMARY_LOC = MAIN_LOC+'scratch/HSMmodel/TFtrainingSummary/subsampling/'
    code_for_this_run = "%s_Region%s_numLGN%g_hlsr%.2f_itr%g_seed%g_trial%g_%s/"%(Code,
                                region_num,
                                num_lgn, 
                                hlsr,
                                iterations,
                                SEED,
                                RESTART_TRIAL,
                                dt_stamp)
    summary_dir = SUMMARY_LOC+code_for_this_run
    save_output_dir = OUTPUT_LOC+code_for_this_run   
    summary_fname = "AntolikRegion"+region_num
    
    try: 
        os.makedirs(summary_dir)
    except OSError:
        if not os.path.isdir(summary_dir):
            raise
    
    try: 
        os.makedirs(save_output_dir)
    except OSError:
        if not os.path.isdir(save_output_dir):
            raise
            
    # Download data from a region
    train_input=np.load(DATA_LOC + '/region' + region_num+'/training_inputs.npy')
    train_set=np.load(DATA_LOC + '/region' + region_num+'/training_set.npy')
    
    raw_vld_set = np.load(DATA_LOC + '/region' + region_num+'/raw_validation_set.npy')
    vldinput_set = np.load(DATA_LOC + '/region' + region_num+'/validation_inputs.npy')
    vld_set = np.load(DATA_LOC + '/region' + region_num+'/validation_set.npy')

    if(NORM_RESPONSE):
        train_set = train_set/(train_set.max()-train_set.min())
    num_neuron = np.shape(train_set)[1]
    # Create tensorflow vars
    MODEL_dtype=tf.float64
    images = tf.placeholder(dtype=MODEL_dtype, shape=[None, train_input.shape[-1]], name='images')
    neural_response = tf.placeholder(dtype=MODEL_dtype, shape=[None, train_set.shape[-1]], name='neural_response') # , shape=)


    #with tf.device(GPUcode):
    with tf.device('/gpu:0'):
      with tf.variable_scope('hsm') as scope:
        # Declare and build model
        hsm = tf_HSM()
        pred_neural_response,l1, lgn_out = hsm.build(images, neural_response, SEED)
    
        # Define loss
        loss = log_likelihood(
          predictions=pred_neural_response,
          targets=neural_response)
          
        # Optimize loss
        train_op=ScipyOptimizerInterface(loss, var_list=tf.trainable_variables(), method='TNC', bounds=hsm.bounds_list, jac=tf.gradients(loss, tf.trainable_variables()) ,options={'maxiter': MAXITER, 'disp':True}) #, 'maxCGit':5}) 

        # Track correlation between YY_hat and YY    
        score = tf.nn.l2_loss(pred_neural_response - neural_response)
        
        # Track the loss and score
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("score", score)

        

    
    # Set up summaries and savre
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.global_variables_initializer(),
     tf.local_variables_initializer()))
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    loss_list, activation_summary_lgn, activation_summary_l1, yhat_std, MSE_list, corr_list = [], [], [], [], [],[]
    
    for idx in range(iterations):
      itr_time=time.time()

      train_op=ScipyOptimizerInterface(loss, var_list=tf.trainable_variables(), method='TNC', bounds=hsm.bounds_list, jac=tf.gradients(loss, tf.trainable_variables()) ,options={'maxiter': MAXITER, 'disp':True})
      train_op.minimize(sess, fetches=[loss, score, pred_neural_response, l1, lgn_out],
          feed_dict={images: train_input, neural_response: train_set})

      loss_value=sess.run(loss, feed_dict={images: train_input, neural_response: train_set}); 
      score_value=sess.run(score,feed_dict={images: train_input, neural_response: train_set}); 
      yhat=sess.run(pred_neural_response,feed_dict={images: train_input, neural_response: train_set}); 
      l1_response=sess.run(l1,feed_dict={images: train_input, neural_response: train_set}); 
      lgn_response=sess.run(lgn_out,feed_dict={images: train_input, neural_response: train_set})
      y_vld=sess.run(pred_neural_response,feed_dict={images: vldinput_set, neural_response: vld_set});  # Validation set
      
      corr=computeCorr(yhat, train_set)    
      #corr=correlate_vectors(yhat, train_set)
      corr[np.isnan(corr)]=0.0
      it_corr = np.mean(corr)
      corr_list += [it_corr]
      loss_list += [loss_value]
      MSE_list += [score_value]
      activation_summary_lgn += [np.mean(lgn_response)]
      activation_summary_l1 += [np.mean(l1_response)]
      yhat_std += [np.std(yhat)]
      if idx % SAVER_SAVE == 0:
        print("=================================================")
        print "  CODE :: " + runcodestr
        print("=================================================")
        saver.save(sess, '%s/%s'%(summary_dir,summary_fname),global_step=idx) 
        #lgn_x=sess.run(hsm.lgn_x, feed_dict={images: train_input, neural_response: train_set}); 
        #var = [v for v in tf.trainable_variables()]
        #current_trained = [sess.run(v, feed_dict={images: train_input, neural_response: train_set}) for v in var]
        
      if(idx==0):
        yhat_1st = yhat
        l1_response_1st=l1_response
        lgn_response_1st=lgn_response
        y_vld_1st=y_vld
      print 'Iteration: %s | LOSS: %f | MSE: %f | Corr: %.5f |STD of yhat: %.5f\n Time ::: %s    Time since start::: %s' % (
          idx,
          loss_value,
          score_value,
          it_corr,
          np.std(yhat),
          time.time()-itr_time,
         time.time()-tt_run_time)
    print "Training complete: Time %s" %(time.time() - tt_run_time)
    
    #save
    if(SAVEdat):
        np.savez('%s/TRdat_%s.npz'%(summary_dir,summary_fname), 
         TR_loss=loss_list, 
         TR_mean_LGNact=activation_summary_lgn, 
         TR_mean_L1act=activation_summary_l1, 
         TR_std_pred_response =yhat_std, 
         TR_MSE=MSE_list, 
         TR_corr=corr_list,
         TR_last_pred_response=yhat,
         TR_last_l1_response=l1_response,
         TR_last_lgn_response=lgn_response,
         TR_1st_pred_response=yhat_1st,
         TR_1st_l1_response=l1_response_1st,
         TR_1st_lgn_response=lgn_response_1st,
         VLD_1st_ypredict=y_vld_1st,
         VLD_ypredict=y_vld,
         CONFIG=CONFIG)
    # check the training
    if(VISUALIZE):
        itr_idx = range(iterations)
        plt.subplot(2, 3, 1)
        plt.plot(itr_idx, loss_list, '-ok')
        plt.title('Loss')
        plt.xlabel('iterations')

        plt.subplot(2, 3, 2)
        plt.plot(itr_idx, MSE_list, '-ob')
        plt.title('MSE')
        plt.xlabel('iterations')

        plt.subplot(2, 3, 3)
        plt.plot(itr_idx, corr_list, '-or')
        plt.title('Mean Correlation')
        plt.xlabel('iterations')

        plt.subplot(2, 3, 4)
        plt.plot(itr_idx, activation_summary_lgn, '-or')
        plt.title('Mean LGN activation')
        plt.xlabel('iterations')

        plt.subplot(2, 3, 5)
        plt.plot(itr_idx, activation_summary_l1, '-ob')
        plt.title('Mean L1 activation')
        plt.xlabel('iterations')

        plt.subplot(2,3, 6)
        plt.plot(itr_idx, yhat_std, '-ok')
        plt.title('std of predicted response')
        plt.xlabel('iterations')

        plt.suptitle("Code: %s max iter=%g , itr = %g\n[%s]"%(runcodestr,MAXITER,iterations,str(datetime.now())))
        plt.show()

    if (PLOT_CORR_STATS):
        pred_act = yhat; responses = train_set
        #hist_of_pred_and_record_response(pred_act,responses)

        corr = computeCorr(yhat, train_set)
        corr[np.isnan(corr)]=0.0
        plot_act_of_max_min_corr(pred_act,responses,corr)
        hist_of_pred_and_record_response(pred_act,responses,cell_id=np.argmax(corr))
        
        pred_act = y_vld; responses = vld_set
        #hist_of_pred_and_record_response(pred_act,responses)

        corr = computeCorr(pred_act, responses)
        corr[np.isnan(corr)]=0.0
        plot_act_of_max_min_corr(pred_act,responses,corr)
        hist_of_pred_and_record_response(pred_act,responses,cell_id=np.argmax(corr))
    print("####################################################")
    for cf in CONFIG :
        print  cf +" = " + str(CONFIG[cf])
    print('Finished everything\n Code::  %s\n Time::  %s '%(runcodestr, time.time() - tt_run_time))

if __name__ == "__main__":
    main()


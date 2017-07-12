import numpy as np
import tensorflow as tf
import param
from tf_HSM_upgraded_SciPy import tf_HSM
import os
import matplotlib.pyplot as plt
from datetime import datetime
import re 
from tensorflow.python import debug 
import time
from visualization import *
import sys
from get_host_path import get_host_path
ScipyOptimizerInterface = tf.contrib.opt.ScipyOptimizerInterface

# CUDA_VISIBLE_DEVICES=2 python tf_HSM_main_MaxFunc_100k_upgraded.py RESTART_TRIAL=3 SEED=3 LR=1e-3 ITERATIONS=100000

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
    plt.show()
def create_bound_list(bounds, num_lgn=9, hlsr=0.2, num_neuron=103):
  num_hidden = int(np.floor(hlsr*num_neuron))
  bounds_list = [bounds['x_pos']]*num_lgn+\
    [bounds['y_pos']]*num_lgn +\
    [bounds['size_center']]*num_lgn+\
    [bounds['size_surround']]*num_lgn+\
    [bounds['center_weight']]*num_lgn+\
    [bounds['surround_weight']]*num_lgn+\
    [bounds['hidden_weights']]*num_lgn*num_hidden +\
    [bounds['hidden_layer_threshold']]*num_hidden +\
    [bounds['output_weights']]*num_hidden*num_neuron+\
    [bounds['output_layer_threshold']]*num_neuron
  return bounds_list
     
#########################################################################
# Main script
########################################################################

def main():
    #########################################################################
    # Simulation Config
    ########################################################################

    REGION =1; RESTART_TRIAL=0; SEED =13; ITERATIONS=100; LR = 1e-3; NUM_LGN=9; HLSR=0.2;
    if len(sys.argv) > 1:
        for ii in range(1,len(sys.argv)):
            arg = sys.argv[ii]
            print(arg)
            exec(arg) 
    tf.set_random_seed(SEED)
    print('SEED : %g'%(SEED))
    tt_run_time = time.time()
    
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
    runcodestr ="Machine: %s Region: %s LR: %.5f Iterations: %g Restart#: %g"%(HOST,region_num, lr, iterations, RESTART_TRIAL)
    SAVER_SAVE = int(iterations/10.0)
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
    'HLSR':HLSR
    }
    
    SUMMARY_DIR = 'TFtrainingSummary/SciPy/'
    #import ipdb; ipdb.set_trace()
    # Download data from a region
    train_input=np.load(PATH + 'Data/region' + region_num+'/training_inputs.npy')
    train_set=np.load(PATH + 'Data/region' + region_num+'/training_set.npy')

    if(NORM_RESPONSE):
        train_set = train_set/(train_set.max()-train_set.min())
    num_neuron = np.shape(train_set)[1]
    # Create tensorflow vars
    images = tf.placeholder(dtype=tf.float32, shape=[None, train_input.shape[-1]], name='images')
    neural_response = tf.placeholder(dtype=tf.float32, shape=[None, train_set.shape[-1]], name='neural_response') # , shape=)


    #with tf.device(GPUcode):
    with tf.device('/gpu:0'):
      with tf.variable_scope('hsm') as scope:
        # Declare and build model
        hsm = tf_HSM()
        pred_neural_response,l1, lgn_out = hsm.build(images, neural_response)
    
        # Define loss
        #loss = tf.contrib.losses.log_loss(
        # predictions=pred_neural_response,
        # targets=neural_response) 
        loss = log_likelihood(
          predictions=pred_neural_response,
          targets=neural_response)
        
        # Optimize loss
        bounds_list = create_bound_list(hsm.bounds, num_lgn=num_lgn, hlsr=hlsr, num_neuron=num_neuron)
        #train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        train_op=ScipyOptimizerInterface(loss, var_list=tf.trainable_variables(), method='TNC', bounds=bounds_list, options={'maxiter': 1000, 'disp':True}) 
        #train_op=ScipyOptimizerInterface(loss, method='L-BFGS-B', bounds=bounds_list, options={'maxiter': 1000, 'disp':True}) 
        #train_op=ScipyOptimizerInterface(loss, options={'maxiter': 100, 'disp':True}) 
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
    
    #train_op.minimize(sess)
    summary_dir = os.path.join(SUMMARY_DIR+"AntolikRegion%s_lr%.5f_itr%g_SEED%g_%s"%(region_num,lr,iterations,SEED,dt_stamp))
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)


    loss_list, activation_summary_lgn, activation_summary_l1, yhat_std, MSE_list, corr_list = [], [], [], [], [],[]
    
    summary_fname = "trainedHSM_region"+region_num+"_trial%g"%(RESTART_TRIAL)
    """
    with tf.Session() as session: 
      train_op.minimize(session)
    #Variables subject to optimization are updated in-place at the end of optimization.
    #Note that this method does not just return a minimization Op, unlike Optimizer.minimize(); instead it actually performs minimization by executing commands to control a Session.
Wrapper allowing `scipy.optimize.minimize` to operate a `tf.Session`.
https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/opt/python/training/external_optimizer.py 
    """
    
    #for idx in range(iterations):
      #import ipdb; ipdb.set_trace()
    itr_time=time.time()
    import ipdb; ipdb.set_trace()
    #loss_value, score_value, yhat, l1_response, lgn_response=train_op.minimize(sess, fetches=[loss, score, pred_neural_response, l1, lgn_out],
     #   feed_dict={images: train_input, neural_response: train_set})

    train_op.minimize(sess, fetches=[score, pred_neural_response, l1, lgn_out],
        feed_dict={images: train_input, neural_response: train_set})
      #_, loss_value, score_value, yhat, l1_response, lgn_response = sess.run(
      #  [train_op, loss, score, pred_neural_response, l1, lgn_out],
      #  feed_dict={images: train_input, neural_response: train_set})
    import ipdb; ipdb.set_trace()
    #with sess.as_default():
      #loss_value=loss.eval(); score_value=score.eval(); yhat=pred_neural_response.eval(); l1_response=l1.eval(); lgn_response=lgn_out.eval();
    loss_value=loss; score_value=score; yhat=pred_neural_response; l1_response=l1; lgn_response=lgn_out

    #corr=computeCorr(yhat, train_set)    
    corr=correlate_vectors(yhat, train_set)
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
        #saver.save(sess, '%s/%s'%(summary_dir,summary_fname),global_step=idx)
    if(idx==0):
      yhat_1st = yhat
      l1_response_1st=l1_response
      lgn_response_1st=lgn_response
    print 'Iteration: %s | Loss: %.5f | MSE: %.5f | Corr: %.5f |STD of yhat: %.5f\n Time ::: %s    Time since start::: %s' % (
        idx,
        loss_value,
        score_value,
        it_corr,
        np.std(yhat),
        time.time()-itr_time,
       time.time()-tt_run_time)
    #import ipdb; ipdb.set_trace() 
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
         CONFIG=CONFIG)
    # check the training
    if(VISUALIZE):
        itr_idx = range(iterations)
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

    if (PLOT_CORR_STATS):
        pred_act = yhat; responses = train_set
        #hist_of_pred_and_record_response(pred_act,responses)

        corr = computeCorr(yhat, train_set)
        corr[np.isnan(corr)]=0.0
        plot_act_of_max_min_corr(pred_act,responses,corr)
        hist_of_pred_and_record_response(pred_act,responses,cell_id=np.argmax(corr))
    print("####################################################")
    for cf in CONFIG :
        print  cf +" = " + str(CONFIG[cf])
    print('Finished everything\n Code::  %s\n Time::  %s '%(runcodestr, time.time() - tt_run_time))

if __name__ == "__main__":
    main()


import numpy as np
import tensorflow as tf
import param
from tf_HSM_MLPonly import tf_HSM
import os
import matplotlib.pyplot as plt
from datetime import datetime
import re 
from tensorflow.python import debug 
import time
from visualization import *

tt_run_time = time.time()
def correlate_vectors(yhat_array, y_array):
  corrs = []
  
  for yhat, y in zip(np.transpose(yhat_array), np.transpose(y_array)):
    tc = np.corrcoef(yhat, y)[1,0]
    if np.isnan(tc):
      tc=0.0
    #tc = (np.isnan(tc) == False).astype(float) * tc
    corrs += [tc]
  return corrs

def get_trained_Ks(Ks, num_LGN=9):
  #Note : add num_lgn and hlsr later  DEFAULT num_LGN = 9 , hlsr = 0.2 --> layers 
  # x,y = center coordinate
  n = num_LGN
  x=Ks[0:n]; 
  i=1; y=Ks[n*i:n*(i+1)]; 
  i=2; sc=Ks[n*i:n*(i+1)]; i=3; ss=Ks[n*i:n*(i+1)]; i=4; rc=Ks[n*i:n*(i+1)]; i=5; rs=Ks[n*i:n*(i+1)];
  x=np.transpose(x); y=np.transpose(y); 
  sc=np.transpose(sc); ss=np.transpose(ss);
  rc=np.transpose(rc); rs=np.transpose(rs);

  return x,y,sc,ss,rc,rs


def get_LGN_out(X,x,y,sc,ss,rc,rs):
  # X = input image 
  img_vec_size=int(np.shape(X)[0])
  img_size = int(np.sqrt(img_vec_size ))
  num_LGN= np.shape(sc)[0]
  
  xx,yy = np.meshgrid(np.arange(img_size),np.arange(img_size))
  xx = np.reshape(xx,[img_vec_size]); yy = np.reshape(yy,[img_vec_size]);
  
  #Below
  lgn_kernel = lambda i,x,y,sc,ss,rc,rs: np.dot(X, ((rc[i]*(np.exp(-((xx- x[i])**2 + (yy - y[i])**2)/2/sc[i]).T/(2*sc[i]*np.pi))) - rs[i]*(np.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2 /(sc[i]+ss[i])).T/(2*(sc[i]+ss[i])*np.pi))))
  
  lgn_ker_out = np.ndarray([num_LGN],dtype=float)
  
  for i in np.arange(num_LGN):
    lgn_ker_out[i] = lgn_kernel(i,x,y,sc,ss,rc,rs)

  return lgn_ker_out

def log_likelihood(predictions,targets,epsilon =0.0000000000000000001):
  return tf.reduce_sum(predictions) - tf.reduce_sum(tf.mul(targets,tf.log(predictions + epsilon)))
"""
if self.error_function == 'LogLikelyhood':
   self.model = T.sum(model_output) - T.sum(self.Y * T.log(model_output+0.0000000000000000001))
elif self.error_function == 'MSE':
   self.model = T.sum(T.sqr(model_output - self.Y))

"""

# Main script
########################################################################
dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
        
region_num = '1'
runcodestr ="train MLP only "



# Download data from a region
train_input=np.load('/home/pachaya/AntolikData/SourceCode/Data/region' + region_num+'/training_inputs.npy')
train_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region' + region_num+'/training_set.npy')

#Hyperparameters
NUM_LGN=9; HLSR =0.2
#load trained parameters for DoG
num_LGN = NUM_LGN; hlsr = HLSR;
[trained_Ks,trained_hsm] = np.load('out_region'+region_num+'.npy')
x,y,sc,ss,rc,rs = get_trained_Ks(trained_Ks,num_LGN)
#import ipdb; ipdb.set_trace()
# Create tensorflow vars
images = tf.placeholder(dtype=tf.float32, shape=[None, train_input.shape[-1]], name='images')
neural_response = tf.placeholder(dtype=tf.float32, shape=[None, train_set.shape[-1]], name='neural_response') 
trained_x = tf.placeholder(dtype=tf.float32, shape=[num_LGN], name='x_position') 
trained_y = tf.placeholder(dtype=tf.float32, shape=[num_LGN], name='y_position') 
trained_sc = tf.placeholder(dtype=tf.float32, shape=[num_LGN], name='size_center') 
trained_ss = tf.placeholder(dtype=tf.float32, shape=[num_LGN], name='size_surround') 
trained_rc = tf.placeholder(dtype=tf.float32, shape=[num_LGN], name='center_weight') 
trained_rs = tf.placeholder(dtype=tf.float32, shape=[num_LGN], name='surround_weight') 


lr = 1e-3
iterations = 2500




########################################################################


with tf.device('/gpu:2'):
  with tf.variable_scope('hsm') as scope:
    # Declare and build model
    hsm = tf_HSM()
    pred_neural_response,l1, lgn_out = hsm.build(images, neural_response,
      x=trained_x, y=trained_y,
      sc=trained_sc, ss=trained_ss,
      rc=trained_rc, rs=trained_rs)

    # Define loss
    #loss = tf.contrib.losses.log_loss(
    # predictions=pred_neural_response,
    # targets=neural_response) 
    loss = log_likelihood(
      predictions=pred_neural_response,
      targets=neural_response)

    # Optimize loss
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    
    # Track correlation between YY_hat and YY    
    score = tf.nn.l2_loss(pred_neural_response - neural_response)
    
    # Track the loss and score
    tf.scalar_summary("loss", loss)
    tf.scalar_summary("score", score)


# Set up summaries and saver
saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)
summary_op = tf.merge_all_summaries()

# Initialize the graph
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

# Need to initialize both of these if supplying num_epochs to inputs
sess.run(tf.group(tf.initialize_all_variables(),
 tf.initialize_local_variables()))
summary_dir = os.path.join(  "TFtrainingSummary/"  'AntolikRegion'+region_num+'_' + dt_stamp)#declare a directory to store summaries here!
   #config.train_summaries, config.which_dataset + '_' + dt_stamp)
summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)


loss_list, activation_summary_lgn, activation_summary_l1, yhat_std, MSE_list, corr_list = [], [], [], [], [],[]
for idx in range(iterations):
  
  _, loss_value, score_value, yhat, l1_response, lgn_response = sess.run(
    [train_op, loss, score, pred_neural_response, l1, lgn_out],
    feed_dict={images: train_input, neural_response: train_set,
    trained_x: x, trained_y: y,
    trained_sc: sc, trained_ss: ss,
    trained_rc: rc, trained_rs: rs,
    })
  #it_corr = np.mean(correlate_vectors(yhat, train_set))
 
  corr=computeCorr(yhat, train_set)
  corr[np.isnan(corr)]=0.0
  it_corr = np.mean(corr)
  
  corr_list += [it_corr]
  loss_list += [loss_value]
  MSE_list += [score_value]
  activation_summary_lgn += [np.mean(lgn_response)]
  activation_summary_l1 += [np.mean(l1_response)]
  yhat_std += [np.std(yhat)]
  saver.save(sess, 'save_trained_HSM')
  
  print 'Iteration: %s | Loss: %.5f | MSE: %.5f | Corr: %.5f |STD of yhat: %.5f' % (
    idx,
    loss_value,
    score_value,
    it_corr,
    np.std(yhat))


print "Training complete: Time %s" %(time.time() - tt_run_time)
# Analyze the results
#Visualization
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

from scipy.stats import pearsonr

# load trained model 
#with tf.Session() as sess:    
def hist_of_pred_and_record_response(pred_response, recorded_response, cell_id=0):
  plt.subplot(121); plt.hist(recorded_response[:,cell_id]); plt.title('Recorded Response');
  plt.subplot(122); plt.hist(yhat[:,cell_id]); plt.title('Predicted Response');
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


if (True):
    import ipdb; ipdb.set_trace()
    from visualization import *
    
    pred_act = yhat; responses = train_set
    corr = computeCorr(yhat, train_set)
    
    plot_act_of_max_min_corr(pred_act,responses,corr)
    hist_of_pred_and_record_response(pred_act,responses,cell_id=np.argmax(corr))
    import ipdb; ipdb.set_trace()    

    
#saver = tf.train.import_meta_graph('save_trained_HSM.meta')
#saver.restore(sess,tf.train.latest_checkpoint('./'))

print(runcodestr)

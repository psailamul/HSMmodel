import numpy as np
import tensorflow as tf
import param
from tf_HSM import tf_HSM
import os


def correlate_vectors(yhat_array, y_array):
  corrs = []
  for yhat, y in zip(yhat_array, y_array):
    tc = np.corrcoef(yhat, y)[0]
    tc = (np.isnan(tc) == False).astype(float) * tc
    corrs += [tc]
  return np.concatenate(corrs)

def get_trained_Ks(Ks, num_LGN=9):
  #Note : add num_lgn and hlsr later  DEFAULT num_LGN = 9 , hlsr = 0.2 --> layers 
  # x,y = center coordinate
  n = num_LGN
  x=Ks[0:n]; 
  i=1; y=Ks[n*i:n*(i+1)]; 
  i=2; sc=Ks[n*i:n*(i+1)]; i=3; ss=Ks[n*i:n*(i+1)]; i=4; rc=Ks[n*i:n*(i+1)]; i=5; rs=Ks[n*i:n*(i+1)];
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


# Main script
########################################################################
dt_stamp = '17_06_06' 
region_num = '1'
########################################################################

# Download data from a region
train_input=np.load('/home/pachaya/AntolikData/SourceCode/Data/region' + region_num+'/training_inputs.npy')
train_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region' + region_num+'/training_set.npy')

#load trained LGN hyperparameters
#num_LGN = NUM_LGN; hlsr = HLSR;
#[trained_Ks,trained_hsm] = np.load('out_region'+region_num+'.npy')


# Create tensorflow vars
images = tf.placeholder(dtype=tf.float32, shape=[None, train_input.shape[-1]], name='images')
neural_response = tf.placeholder(dtype=tf.float32, shape=[None, train_set.shape[-1]], name='neural_response') # , shape=)
lr = 1e-3
iterations = 1000


#load trained parameters for DoG
#x,y,sc,ss,rc,rs = get_trained_Ks(Ks,9)
#import pdb; pdb.set_trace()

with tf.device('/gpu:0'):
  with tf.variable_scope('hsm') as scope:
    # Declare and build model
    hsm = tf_HSM()
    pred_neural_response = hsm.build(images, neural_response)

    # Define loss
    loss = tf.contrib.losses.log_loss(
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
   # config.train_summaries, config.which_dataset + '_' + dt_stamp)
summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)


corr_list = []
for idx in range(iterations):
  _, loss_value, score_value, yhat = sess.run(
    [train_op, loss, score, pred_neural_response],
    feed_dict={images: train_input, neural_response: train_set})
  it_corr = correlate_vectors(yhat, train_set).mean()
  corr_list += [it_corr]
  print 'Iteration: %s | Loss: %.5f | MSE: %.5f | Corr: %.5f' % (
    idx,
    loss_value,
    score_value,
    it_corr)
# CNN for Antolik data

import numpy as np
import tensorflow as tf
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
# CUDA_VISIBLE_DEVICES=2 python 
import shelve

def save_object(obj, filename):
    """Pkl object."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Un-pkl object."""
    with open(filename, 'rb') as inputfile:
        return pickle.load(inputfile)

#HSM_dir = 'HSM/Data/region1/'
SAVE_dir = 'run_region2/'
HSM_dir = '/home/pachaya/HSMmodel/Data/region2/'

images=np.load(os.path.join(HSM_dir,'training_inputs.npy'))
neuact=np.load(os.path.join(HSM_dir,'training_set.npy'))

testset_images=np.load(os.path.join(HSM_dir,'validation_inputs.npy'))
testset_neuact=np.load(os.path.join(HSM_dir,'validation_set.npy'))


num_neuron = neuact.shape[1]
num_images = images.shape[0]
img_size = 31
batch_size = 1
VLD_prob = 0.1
# injected noise strength
sigma = 0.1

# l2_weight_regularization for every layer
l2_weight = 1e-3
l1_act = 1e-3

#Learning rate
lr = 1e-4

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def weight_variable(in_shape):
  #initial = tf.truncated_normal(shape=in_shape, stddev=0.1)
  initial = tf.random_normal(shape=in_shape, mean=0.0, stddev=sigma, dtype=tf.float32) 
  
  return tf.Variable(initial)

def bias_variable(in_shape):
  #initial = tf.constant(0.1, shape=shape)
  initial = tf.random_normal(shape=in_shape, mean=0.0, stddev=sigma, dtype=tf.float32) 

  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME') #Default in Keras is VALID

def batch_flatten(x):
    """Turn a nD tensor into a 2D tensor with same 0th dimension.
    In other words, it flattens each data samples of a batch.
    # Arguments
        x: A tensor or variable.
    # Returns
        A tensor.
    """
    x = tf.reshape(x, tf.stack([-1, tf.reduce_prod(tf.shape(x)[1:])]))
    return x

def parametric_softplus(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  betas = tf.get_variable('beta', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)

  return alphas * tf.log(1 + tf.exp(betas * _x)) # Equation from https://faroit.github.io/keras-docs/0.3.3/layers/advanced_activations/


#Deep retina model
with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, shape = [None, img_size, img_size]) # placeholder for input images
    y_ = tf.placeholder(tf.float32, shape = [None, num_neuron]) # placeholder for true labels for input images
    #b_holder = 
    conv1_numfeat =8
    conv1_size = 9

    W_conv1 = weight_variable([conv1_size, conv1_size, batch_size, conv1_numfeat]) 
    b_conv1 = bias_variable([conv1_numfeat])
    x_image = tf.reshape(x, [-1, img_size, img_size, batch_size])

    l_conv1 =conv2d(x_image, W_conv1) + b_conv1
    g_conv1 = gaussian_noise_layer(l_conv1, sigma) #+ gaussian noise
    h_conv1 = tf.nn.relu(g_conv1) 

    h_pool1 = max_pool_2x2(h_conv1) 

    conv2_numfeat = 16
    conv2_size = 15

    W_conv2 = weight_variable([conv2_size, conv2_size, conv1_numfeat, conv2_numfeat])
    b_conv2 = bias_variable([conv2_numfeat])


    l_conv2 =conv2d(h_pool1, W_conv2) + b_conv2 
    g_conv2 = gaussian_noise_layer(l_conv2, sigma) #+ gaussian noise
    h_conv2 = tf.nn.relu(g_conv2)  

    h_pool2 = max_pool_2x2(h_conv2) 


    img_size_after_conv2 = 8
    imsz = img_size_after_conv2

    #Flatten 
    flatten = batch_flatten(h_pool2)

    #Dense
    W_fc1 = weight_variable([imsz * imsz * conv2_numfeat, num_neuron])
    b_fc1 = bias_variable([num_neuron])

    h_pool2_flat = tf.reshape(h_pool2, [-1, imsz*imsz*conv2_numfeat])
    h_fc1 = parametric_softplus(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    loss = tf.nn.log_poisson_loss(y_, h_fc1)
    loss = tf.reduce_mean(loss)

    # Loss function with Regularization 
    w_regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2)
    activity_regularizers = tf.abs(h_fc1)
    reg = tf.reduce_mean(l2_weight * w_regularizers+l1_act*activity_regularizers)
    loss_fn = tf.reduce_mean(loss + l2_weight * w_regularizers+l1_act*activity_regularizers)

    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_fn)
    """
    loss = tf.reduce_sum(loss)
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    """
#to do : check accuracy with validation 

def mse_score(y_pred, y_true,ax=None):
    return ((y_true - y_pred) ** 2).mean(axis=ax)

def pearson_corr_score(y_pred, y_true):
    return np.corrcoef(y_true,y_pred)[1,0]

def pearson_corr_Zscore(y_pred, y_true):
    Zpred = np.true_divide(y_pred - np.mean(y_pred), np.std(y_pred))
    Ztrue = np.true_divide(y_true - np.mean(y_true), np.std(y_true))
    return np.corrcoef(Ztrue,Zpred)[1,0]

def batch_train_vld(img, act, prob = 0.1):
    num_im = img.shape[0]
    num_vld = num_im*prob #180 for p=0.1
    shuff_id = range(num_im)
    np.random.shuffle(shuff_id)
    img_shuff = img[shuff_id,:]
    act_shuff = act[shuff_id,:]
    vld_img = img_shuff[:num_vld,:]
    vld_act = act_shuff[:num_vld,:]
    train_img = img_shuff[num_vld:,:]
    train_act = act_shuff[num_vld:,:]
    return train_img, train_act, vld_img, vld_act

record_ll =[]
record_lreg =[]
record_lfn =[]
record_mse=[]
record_r=[]


vld_mse=[]
vld_r=[]
vld_r_zscore = []

num_epoch=1000

start = time.time()


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  #TO DO: add epochs and shuffle images
  if batch_size == num_images:
    for jj in range(num_epoch):
      batch_im = images
      batch_im = np.reshape(batch_im,[-1,img_size,img_size])
      batch_res = neuact
      feedDict = {x:batch_im, y_:batch_res}
      fetch = [train_step, loss_fn, loss, reg, h_fc1]
      train_op, lfn,lo, lreg, out = sess.run(fetch, feed_dict=feedDict)
      mse = mse_score(out, batch_res)
      rr = pearson_corr_score(out, batch_res)
      record_ll.append(lo)
      record_lreg.append(lreg)
      record_lfn.append(lfn)
      record_mse.append(mse)
      record_r.append(rr)
      print "j:",jj," loss:",lo," reg",lreg," loss_fn",lfn,' mse:',mse,' r:',rr
      print "Time lapse %s"%(time.time()-start)
      if jj%50 == 0:
        import ipdb; ipdb.set_trace()
  else:
    for jj in range(num_epoch):
      train_img, train_act, vld_img, vld_act = batch_train_vld(img=images, act=neuact, prob = VLD_prob)
      num_train = train_img.shape[0]
      num_vld = vld_img.shape[0]
      for i in np.arange(0,num_train-batch_size, batch_size):
        batch_im = train_img[i:i+batch_size,:]
        batch_im = np.reshape(batch_im,[-1,img_size,img_size])
        batch_res = train_act[i:i+batch_size,:]
        feedDict = {x:batch_im, y_:batch_res}
        fetch = [train_step, loss_fn, loss, reg, h_fc1]
        train_op, lfn,lo, lreg, out = sess.run(fetch, feed_dict=feedDict)
        mse = mse_score(out, batch_res)
        rr = pearson_corr_score(out, batch_res)
        record_ll.append(lo)
        record_lreg.append(lreg)
        record_lfn.append(lfn)
        record_mse.append(mse)
        record_r.append(rr)
        #print "j:",jj,' i:',i," loss:",lo," reg",lreg," loss_fn",lfn,' mse:',mse,' r:',rr
      mse_s = []
      rr_s = []
      rrz_s = []
      for img, act in zip(vld_img, vld_act):
        img = np.reshape(img,[-1,img_size,img_size])
        act = np.reshape(act,[-1,num_neuron])
        feedDict = {x:img, y_:act}
        act_pred = sess.run(h_fc1, feed_dict=feedDict)
        mse_s.append(mse_score(act_pred, act))
        rr_s.append(pearson_corr_score(act_pred, act))
        rrz_s.append(pearson_corr_Zscore(act_pred, act))

      mean_mse=np.mean(mse_s)
      mean_rr =np.mean(rr_s)
      mean_rrz=np.mean(rrz_s)

      vld_mse.append(mean_mse)
      vld_r.append(mean_rr)
      vld_r_zscore.append(mean_rrz)

      print "j:",jj,' mse:',mean_mse,' r:',mean_rr,' r of Z:',mean_rrz
      print "Time lapse %s"%(time.time()-start)
      if (jj+1)%50 == 0 or jj == 0:
        #import ipdb; ipdb.set_trace()
        max_vld = np.argmax(rrz_s)
        img =vld_img[max_vld,:]
        img = np.reshape(img,[-1,img_size,img_size])
        act = vld_act[max_vld,:]
        act = np.reshape(act,[-1,num_neuron])
        feedDict = {x:img, y_:act}
        act_pred = sess.run(h_fc1, feed_dict=feedDict)
        act_pred = np.reshape(act_pred, [num_neuron,])
        act = np.reshape(act, [num_neuron,])

        Zpred = np.true_divide(act_pred - np.mean(act_pred), np.std(act_pred))
        Ztrue = np.true_divide(act - np.mean(act), np.std(act))

        fig, ax = plt.subplots( nrows=1, ncols=1 ) 
        ax.plot(range(num_neuron), Ztrue,'-ok') #true
        ax.plot(range(num_neuron), Zpred,'-or') #pred
        ax.set_title("Image with maximum r(z-score) in validation set, r=%g"%(rrz_s[max_vld]))
        ax.set_xlabel("Neuron ID")
        ax.set_ylabel("Activity (red: predict, black:true)")
        #plt.show()
        fname = os.path.join(SAVE_dir,"Epoch_%g_best_r_z.png"%(jj))
        plt.savefig(fname)
        plt.close(fig)
        saver = tf.train.Saver()
        saver.save(sess,os.path.join(SAVE_dir,"Epoch%g"%(jj)))
  import ipdb; ipdb.set_trace()
  #Get the prediction for test set 
  saver = tf.train.Saver()
  saver.save(sess,os.path.join(SAVE_dir,"LastEpoch_%g"%(jj)))
  test_mse_s = []
  test_rr_s = []
  test_rrz_s = []
  test_layers =[]
  test_predict =[]
  for img, act in zip(testset_images, testset_neuact):
    img = np.reshape(img,[-1,img_size,img_size])
    act = np.reshape(act,[-1,num_neuron])
    feedDict = {x:img, y_:act}
    fetch = [h_conv1, h_pool1, h_conv2, h_pool2, h_fc1]
    hcv1, hp1, hcv2, hp2, act_pred = sess.run(fetch, feed_dict=feedDict)
    test_predict.append(act_pred)
    test_layers.append([hcv1, hp1, hcv2, hp2])
    test_mse_s.append(mse_score(act_pred, act))
    test_rr_s.append(pearson_corr_score(act_pred, act))
    test_rrz_s.append(pearson_corr_Zscore(act_pred, act))
  outfile = os.path.join(SAVE_dir,"save_data")
  all_data = shelve.open(outfile,'c') 
  all_data['test_layers']=test_layers 
  all_data['test_mse_s']=test_mse_s
  all_data['test_rr_s']=test_rr_s
  all_data['test_rrz_s']=test_rrz_s
  all_data['test_predict']=test_predict
 
  all_data.close()
  print "Finish Session"

  #plot mean<mse, r, rz> per epoch to show that it's learned
import ipdb; ipdb.set_trace()
print "Done"


import shelve
filename='~/HSMmodel/shelve_all.out'
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()
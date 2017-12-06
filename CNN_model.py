# CNN for Antolik data

import numpy as np
import tensorflow as tf
import os


HSM_dir = 'HSM/Data/region1/'

#HSM_dir = '/home/pachaya/HSMmodel/Data/region1/'

images=np.load(os.path.join(HSM_dir,'training_inputs.npy'))
neuact=np.load(os.path.join(HSM_dir,'training_set.npy'))


num_neuron = neuact.shape[1]
num_images = images.shape[0]
img_size = 31
batch_size = 1

# injected noise strength
sigma = 0.1

# convolutional layer sizes
convlayers = [(16, 15), (8, 9)] # (n, size)

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
with tf.device('/device:GPU:1'):
    x = tf.placeholder(tf.float32, shape = [None, img_size, img_size]) # placeholder for input images
    y_ = tf.placeholder(tf.float32, shape = [None, num_neuron]) # placeholder for true labels for input images

    import ipdb; ipdb.set_trace()
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
    # + W reg

    l_conv2 =conv2d(h_pool1, W_conv2) + b_conv2 # ? 1 1 16
    g_conv2 = gaussian_noise_layer(l_conv2, sigma) #+ gaussian noise
    h_conv2 = tf.nn.relu(g_conv2)  # ? 1 1 16
    # + act reg

    h_pool2 = max_pool_2x2(h_conv2) #img size = 1


    img_size_after_conv2 = 8
    imsz = img_size_after_conv2

    #Flatten 
    flatten = batch_flatten(h_pool2)

    #Dense
    W_fc1 = weight_variable([imsz * imsz * conv2_numfeat, num_neuron])
    b_fc1 = bias_variable([num_neuron])

    h_pool2_flat = tf.reshape(h_pool2, [-1, imsz*imsz*conv2_numfeat])
    h_fc1 = parametric_softplus(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    loss = tf.nn.log_poisson_loss(targets = y_, log_input=h_fc1)
    loss = tf.reduce_sum(loss)

    # Loss function with Regularization 
    w_regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2)
    activity_regularizers = tf.abs(h_fc1)
    loss_fn = tf.reduce_sum(loss + l2_weight * w_regularizers+l1_act*activity_regularizers)


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

record_l =[]
record_l_reg =[]
record_mse=[]
record_r=[]
num_epoch=100

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  #TO DO: add epochs and shuffle images
  for jj in range(num_epoch):
      for i in np.arange(0,num_images-batch_size, batch_size):
        batch_im = images[i:i+batch_size,:]
        batch_im = np.reshape(batch_im,[-1,img_size,img_size])
        batch_res = neuact[i:i+batch_size,:]
        feedDict = {x:batch_im, y_:batch_res}
        #fetch = [train_step, loss, h_fc1]
        #train_op, l, out = sess.run(fetch, feed_dict=feedDict)
        #l_reg='N/A'
        fetch = [train_step, loss_fn, loss, h_fc1]
        train_op, l_reg,l, out = sess.run(fetch, feed_dict=feedDict)
        record_l_reg.append(l_reg)

        mse = mse_score(out, batch_res)
        rr = pearson_corr_score(out, batch_res)
        record_l.append(l)
        record_mse.append(mse)
        record_r.append(rr)
        
        print "i:",i," loss:",l," loss_reg",l_reg,' mse:',mse,' r:',rr
      if jj%50 == 0:
        import ipdb; ipdb.set_trace()
    

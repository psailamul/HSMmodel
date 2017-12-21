import numpy as np
import tensorflow as tf
import os
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import shelve
# CUDA_VISIBLE_DEVICES=2 python 

#Data from Antolik et al, 2016 http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927 
region = 1 #There're 3 recording regions (all in primary visual cortex). Each trained separately
HSM_dir = "Data/region%g/"%(region) 
SAVE_dir = "TFtrainingSummary/HSMrebuild/run_region%g"%(region)


images=np.load(os.path.join(HSM_dir,'training_inputs.npy'))
neuact=np.load(os.path.join(HSM_dir,'training_set.npy'))

testset_images=np.load(os.path.join(HSM_dir,'validation_inputs.npy'))
testset_neuact=np.load(os.path.join(HSM_dir,'validation_set.npy'))

# number of input images and number of recorded neurons
num_neuron = neuact.shape[1] 
num_images = images.shape[0]
img_size = 31 
batch_size = 1 #Hyperparameters
VLD_prob = 0.1 #Percent of data treated as validation


def DoG(self, x, y, sc, ss, rc, rs):
  # Passing the parameters for a LGN neuron
  pi = tf.constant(np.pi, dtype=self.model_dtype)
  pos = ((self.grid_xx - x)**2 + (self.grid_yy - y)**2)
  center = tf.exp(-pos/2/sc) / (2*(sc)*pi)
  surround = tf.exp(-pos/2/(sc + ss)) / (2*(sc + ss)*pi)
  weight_vec = tf.reshape((rc*(center)) - (rs*(surround)), [-1, 1])
  return tf.matmul(self.images, weight_vec)


def add_lgn_cell(img_size):
    """
    Return a tensor for a LGN cell
    """
    #x, y, sc, ss, rc, rs
    xx, yy = np.meshgrid(img_size,img_size)
    grid_xx = tf.constant(xx)
    grid_yy = tf.constant(yy)

    
    pi = tf.constant(np.pi)
    pos = ((grid_xx - x)**2 + (self.grid_yy - y)**2)
    center = tf.exp(-pos/2/sc) / (2*(sc)*pi)
    surround = tf.exp(-pos/2/(sc + ss)) / (2*(sc + ss)*pi)
    
    weight_vec = tf.reshape((rc*(center)) - (rs*(surround)), [-1, 1])
    return tf.matmul(self.images, weight_vec)


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

""" 
Outline 

1. Input image
2. Each image send to LGN
3. Each LGN take inmage then use DoG to extract feature from an image and send out
4. Output from all LGN cells become  input to the MLP layer (sum responses and send as input to layer 1) 
Note: Their doesn't seem to sum though. Only concatenate so check again ?

5. LGNout --- connected to hidden layer (W, no bias)

6.Hidden layer : W[#lgn,#hidden] , no bias , activation fn = logistic-loss type transfer functions --- "shifted relu" 
May be possible to add "threshold" with input to each hidde layer cell 
---> check bound for the threshold 
---> use max fn? to check positive and negative part (x-t) > 0 and x-t <0
---> size of threshold = # of hidden
hidden = W*(LGN + threshold)
hidden_out = relu(hidden)

7. Output layer : Wout [ #hidden, #out] 
out = Wout * (hidden_out + threshold2)


 The first layer consists of linear kernels of LGN units that are modeled as 2D difference-of-Gaussians functions 
 (see Fig 2). 

 Units in the second layer sum the responses of LGN-like linear units, 
 and pass on the resulting potential via a logistic-loss non-linearity. 
 In this way units in the second layer construct oriented RFs through feed-forward summation 
 of thalamocortical inputs [27]. 


 Linear summation coupled with logistic-loss non-linearity is repeated again in the third layer, 
 which enables construction of RFs that are tuned to orientation but can be insensitive to spatial phase 
 (i.e. units resembling complex cells). 


The log-loss function approximates a linear function with the slope of 1 as (x − t)→∞.






"""
import numpy as np
import tensorflow as tf
import param
from numpy.random import rand, seed


      
def logistic_loss(x,t=0.0, coef=1.0):
  return coef * tf.log(1 + tf.exp(coef*(x-t)))
  
class tf_HSM():
    """
    Model for fitting vision data to neural recordings.

    params:

      images: training input, 
      neural_response: training recorded neural response

    Model:

      images-> LGN -> 2-layer MLP
      LGN: Any vision feature extractor. This model uae different of gaussian (DoG) here


    Output:
      train_op : Optimizer from Tensor Flow 
      loss: loss value. This model use log_loss
      score : MSE
      pred_neural_response: prediction for neural response
    """
    
    def __init__(self, NUM_LGN=9,HLSR=0.2, **params): #def __init__(**params):
      #get_bounds_for_init = lambda lo, up:  [lo + (up-lo)/4.0 , lo + (up-lo)/4.0 + (up-lo)/2.0]
      self.num_lgn=NUM_LGN
      self.hlsr = HLSR
      self.activation = lambda x, y: logistic_loss(x, t=y, coef=1)
      self.images = None
      self.neural_response = None
      self.SEED=None
      self.lgn_trainable=True
      self.LGN_params={}
      self.bounds = {}
      self.bounds_list=[]
      self.init_bounds={}
      self.init_bounds_list=[]
      self.initialized_value_list=[]
      self.initializers_all_params={}
      self.model_dtype=tf.float64

      
    def construct_free_params(self):
      
      # LGN initialization
      self.lgn_x = tf.get_variable(name="x_pos", dtype=self.model_dtype, initializer=self.initializers['x_pos'], trainable=self.lgn_trainable);
      self.lgn_y = tf.get_variable(name="y_pos", dtype=self.model_dtype,  initializer=self.initializers['y_pos'],  trainable=self.lgn_trainable);
      self.lgn_sc = tf.get_variable(name="size_center",dtype=self.model_dtype,   initializer=self.initializers['size_center'],  trainable=self.lgn_trainable) ; 
      self.lgn_ss = tf.get_variable(name="size_surround",dtype=self.model_dtype,  initializer=self.initializers['size_surround'],  trainable=self.lgn_trainable) ; 
      self.lgn_rc = tf.get_variable(name="center_weight", dtype=self.model_dtype,  initializer=self.initializers['center_weight'],  trainable=self.lgn_trainable) ;
      self.lgn_rs = tf.get_variable(name="surround_weight", dtype=self.model_dtype,  initializer=self.initializers['surround_weight'],  trainable=self.lgn_trainable) ;

      # MLP
      self.hidden_w = tf.get_variable(
        name="hidden_weights",
        dtype=self.model_dtype, 
        initializer=self.initializers['hidden_weights'],
        trainable=True)

       
      self.hl_tresh = tf.get_variable(
        name="hidden_layer_threshold",
        dtype=self.model_dtype, 
        initializer=self.initializers['hidden_layer_threshold'],
        trainable=True)

      
      self.output_w = tf.get_variable(
        name="output_weights",
        dtype=self.model_dtype, 
        initializer=self.initializers['output_weights'],
        trainable=True)
      
      self.ol_tresh = tf.get_variable(
        name="output_layer_threshold", 
        dtype=self.model_dtype, 
        initializer=self.initializers['output_layer_threshold'],
        trainable=True)





    def DoG(self, x, y, sc, ss, rc, rs):
      # Passing the parameters for a LGN neuron
      pi = tf.constant(np.pi, dtype=self.model_dtype)
      pos = ((self.grid_xx - x)**2 + (self.grid_yy - y)**2)
      center = tf.exp(-pos/2/sc) / (2*(sc)*pi)
      surround = tf.exp(-pos/2/(sc + ss)) / (2*(sc + ss)*pi)
      weight_vec = tf.reshape((rc*(center)) - (rs*(surround)), [-1, 1])
      return tf.matmul(self.images, weight_vec)

    def LGN(self, i, x_pos, y_pos, lgn_sc, lgn_ss, lgn_rc, lgn_rs):
      output = self.DoG(
          x=x_pos[i],
          y=y_pos[i],
          sc=lgn_sc[i],
          ss=lgn_ss[i],
          rc=lgn_rc[i],
          rs=lgn_rs[i])
      i+=1 
      return i, output

    def LGN_loop(self,x_pos, y_pos, lgn_sc, lgn_ss, lgn_rc, lgn_rs):
        output = []
        
        for i in np.arange(self.num_lgn):
          output += [self.DoG(
              x=x_pos[i],
              y=y_pos[i],
              sc=lgn_sc[i],
              ss=lgn_ss[i],
              rc=lgn_rc[i],
              rs=lgn_rs[i])]
        return tf.concat(axis=1, values=output)
    
    def cond(self, i, x, y, sc, ss, rc, rs):
      return i < self.num_lgn 
   

      
    def create_bounds_list(self, input_bounds):
      bounds=input_bounds
      num_lgn=self.num_lgn
      hlsr=self.hlsr
      num_neuron=self.num_neurons
      num_hidden = self.num_hidden
      
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

      
    def initializer_setting(self):
      #LGN
      self.bounds['x_pos']=(0,31); self.init_bounds['x_pos']=(0,31) # 0-31 
      self.bounds['y_pos']=(0,31); self.init_bounds['y_pos']=(0,31)  # 0-31
      self.bounds['size_center']=(0.1,31); self.init_bounds['size_center']=(0.1,31)   #0.1 - 31
      self.bounds['size_surround']=(0.0,31); self.init_bounds['size_surround']=(0.0,31)  #0.0 - 31
      self.bounds['center_weight']=(0.0,10.0); self.init_bounds['center_weight']=(0.0,10.0)   #0-10
      self.bounds['surround_weight']=(0.0,10.0); self.init_bounds['surround_weight']=(0.0,10.0)   #0-10
      #Hidden Layer
      self.bounds['hidden_weights']=(None,None); self.init_bounds['hidden_weights']=(-10.0,10.0); #init_bounds  #-10, 10 #bounds = (-inf,+inf)
      self.bounds['hidden_layer_threshold']=(0,None); self.init_bounds['hidden_layer_threshold']=(0.0,10.0)#init_bounds # 0-10 #bounds=(0,inf)
      #Output Layer
      self.bounds['output_weights']=(None,None); self.init_bounds['output_weights']=(-10.0, 10.0) # init_bound -10, 10   #bounds = (-inf,+inf)
      self.bounds['output_layer_threshold']=(0,None); self.init_bounds['output_layer_threshold']=(0.0, 10.0) # init_bound 0,10 #bounds = (0, +inf)
      
      self.bounds_list = self.create_bounds_list(self.bounds)
      self.init_bounds_list = self.create_bounds_list(self.init_bounds)
      
      #seed
      seed(self.SEED)
      self.initialized_value_list = [a[0] + (a[1]-a[0])/4.0 + rand()*(a[1]-a[0])/2.0  for a in self.init_bounds_list] #Distribution of rand() = uniform

      
    def create_tf_initializers(self):
      assert self.initialized_value_list is not None 
      init_val_copy=self.initialized_value_list
      initializers ={}
      num_lgn=self.num_lgn
      hlsr=self.hlsr
      num_neuron=self.num_neurons
      num_hidden = self.num_hidden
      #LGN
      keys_list = ('x_pos','y_pos', 'size_center','size_surround','center_weight','surround_weight')
      idx=0
      for kk in keys_list:
        cut_init_value = init_val_copy[idx:idx+num_lgn]; idx+=num_lgn
        initializers[kk] = tf.constant(cut_init_value, shape=[self.num_lgn], dtype=self.model_dtype)
     
      #hidden layer
      cut_init_value = init_val_copy[idx:idx+num_lgn*num_hidden]; idx+=num_lgn*num_hidden
      cut_init_value=np.reshape(cut_init_value, (num_lgn,num_hidden))
      initializers['hidden_weights'] = tf.constant(cut_init_value, shape=(num_lgn,num_hidden), dtype=self.model_dtype)
      
      cut_init_value = init_val_copy[idx:idx+num_hidden]; idx+=num_hidden
      initializers['hidden_layer_threshold'] = tf.constant(cut_init_value, shape=[self.num_hidden], dtype=self.model_dtype)
      
      #output layer
      cut_init_value = init_val_copy[idx:idx+num_hidden*num_neuron]; idx+=num_hidden*num_neuron
      cut_init_value=np.reshape(cut_init_value, (num_hidden,num_neuron))
      initializers['output_weights'] = tf.constant(cut_init_value, shape=(num_hidden,num_neuron), dtype=self.model_dtype)
      cut_init_value = init_val_copy[idx:idx+num_neuron]; idx+=num_neuron
      initializers['output_layer_threshold'] = tf.constant(cut_init_value, shape=[self.num_neurons], dtype=self.model_dtype)
      
      self.initializers = initializers
      
      

    def build(self, data, label, SEED=13):
      self.SEED=SEED
      
      self.img_vec_size = int(data.get_shape()[-1])
      self.img_size = np.sqrt(self.img_vec_size)
      self.num_neurons = int(label.get_shape()[-1])
      self.num_hidden = int(np.floor(self.hlsr*self.num_neurons))
      
      grid_xx, grid_yy = tf.meshgrid(tf.range(self.img_size),tf.range(self.img_size))
      self.grid_xx = tf.cast(tf.reshape(grid_xx, [self.img_vec_size]), dtype=self.model_dtype)
      self.grid_yy = tf.cast(tf.reshape(grid_yy, [self.img_vec_size]), dtype=self.model_dtype)
      
      self.initializer_setting()
      self.create_tf_initializers()

      self.construct_free_params()

      self.images = data
      self.neural_response = label
      assert self.images is not None
      assert self.neural_response is not None

      # DoG
      self.lgn_out = self.LGN_loop(
        x_pos=self.lgn_x,
        y_pos=self.lgn_y,
        lgn_sc=self.lgn_sc,
        lgn_ss=self.lgn_ss,
        lgn_rc=self.lgn_rc,
        lgn_rs=self.lgn_rs)
      # Activation function of LGN itself is linear = > no additional filter
      
      # Run MLP
      self.l1 = self.activation(tf.matmul(self.lgn_out, self.hidden_w), self.hl_tresh) #RELU that shift
      self.output = self.activation(tf.matmul(self.l1, self.output_w), self.ol_tresh)
      
      self.LGN_params={'x_pos':self.lgn_x, 'y_pos':self.lgn_y, 'lgn_sc':self.lgn_sc, 'lgn_ss':self.lgn_ss, 'lgn_rc':self.lgn_rc, 'lgn_rs':self.lgn_rs}
      #self.der=self.der_all()
      return self.output, self.l1, self.lgn_out



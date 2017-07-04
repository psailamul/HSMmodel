import numpy as np
import tensorflow as tf
import param


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
      get_bounds_for_init = lambda lo, up:  [lo + (up-lo)/4.0 , lo + (up-lo)/4.0 + (up-lo)/2.0]
      self.num_lgn=[NUM_LGN]
      self.hlsr = [HLSR]
      self.activation = lambda x, y: logistic_loss(x, t=y, coef=1)
      self.images = None
      self.neural_response = None
      self.lgn_trainable = True
      [L,U]=get_bounds_for_init(-10,10)
      self.UNIFORM_W_init = tf.random_uniform_initializer(L,U)
      [L,U]=get_bounds_for_init(0,10)
      self.UNIFORM_Threshold_init = tf.random_uniform_initializer(L,U)
      self.LGN_params={}
      self.bounds = {}

    def construct_free_params(self):
      # LGN initialization
      self.lgn_x = tf.get_variable(name="x_pos", shape=self.num_lgn, initializer=self.LGN_init, trainable=self.lgn_trainable);self.bounds['lgn_x']=(0,31) # 0-31
      self.lgn_y = tf.get_variable(name="y_pos", shape=self.num_lgn, initializer=self.LGN_init, trainable=self.lgn_trainable);self.bounds['lgn_y']=(0,31)  # 0-31
      self.lgn_sc = tf.get_variable(name="size_center", shape=self.num_lgn, initializer=self.LGN_sc_init, trainable=self.lgn_trainable) ; self.bounds['lgn_sc']=(0.1,31)   #0.1 - 31
      self.lgn_ss = tf.get_variable(name="size_surround", shape=self.num_lgn, initializer=self.LGN_init, trainable=self.lgn_trainable) ; self.bounds['lgn_ss']=(0.0,31)  #0.0 - 31
      self.lgn_rc = tf.get_variable(name="center_weight", shape=self.num_lgn, initializer=self.LGN_init, trainable=self.lgn_trainable) ; self.bounds['lgn_rc']=(0.0,10.0)   #0-10
      self.lgn_rs = tf.get_variable(name="surround_weight", shape=self.num_lgn, initializer=self.LGN_init, trainable=self.lgn_trainable) ; self.bounds['lgn_rs']=(0.0,10.0)  #0-10

      # MLP
      self.hidden_w = tf.get_variable(
        name="hidden_weights",
        shape=(self.num_lgn[0], int(self.num_neurons[0]*self.hlsr[0])), # [9,20]
        initializer=self.UNIFORM_W_init)  #init_bounds  #-10, 10 #bounds = (-inf,+inf)
      self.bounds['hidden_weights']=(None,None)
       
      self.hl_tresh = tf.get_variable(
        name="hidden_layer_threshold",
        shape=int(self.num_neurons[0]*self.hlsr[0]),  #20
        initializer=self.UNIFORM_Threshold_init) #init_bounds # 0-10 #bounds=(0,inf)
      self.bounds['hidden_layer_threshold']=(0,None)
      
      self.output_w = tf.get_variable(
        name="output_weights",
        shape=(int(self.num_neurons[0]*self.hlsr[0]), int(self.num_neurons[0])), #20, 103
        initializer=self.UNIFORM_W_init) # init_bound -10, 10   #bounds = (-inf,+inf)
      self.bounds['output_weights']=(None,None)
      
      self.ol_tresh = tf.get_variable(
        name="output_layer_threshold", #output_layer_threshold
        shape=int(self.num_neurons[0]), #103
        initializer=self.UNIFORM_Threshold_init) # init_bound 0,10 #bounds = (0, +inf)
      self.bounds['output_layer_threshold']=(0,None)

      #Check bounds
      #checkbounds = lambda val, lower_bound, upper_bound : tf.minimum(tf.maximum(val,lower_bound), upper_bound)
      #self.hidden_w =checkbounds(self.hidden_w,-10,10); self.hl_tresh = checkbounds(self.hl_tresh,0,10);
      #self.output_w =checkbounds(self.output_w,-10,10); self.ol_tresh = checkbounds(self.ol_tresh,0,10);
    
    def DoG(self, x, y, sc, ss, rc, rs):
      # Passing the parameters for a LGN neuron
      #import ipdb; ipdb.set_trace()
      #x=14.62563043; y=19.43198948; sc=0.1; ss=1.85884457; rc= 0.45414222; rs=9.79140981;

      #Check bounds
      checkbounds = lambda val, lower_bound, upper_bound : tf.minimum(tf.maximum(val,lower_bound), upper_bound)
      #actually, I would prefer resampling over max/min values

      x = checkbounds(x,0.0,self.img_size); y = checkbounds(y,0.0,self.img_size)
      sc =checkbounds(x,0.1,self.img_size); ss = checkbounds(y,0.0,self.img_size)
      rc=checkbounds(x,0.0,10.0); rs = checkbounds(y,0.0,10.0)

      pi = tf.constant(np.pi)
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
        
        for i in np.arange(self.num_lgn[0]):
          output += [self.DoG(
              x=x_pos[i],
              y=y_pos[i],
              sc=lgn_sc[i],
              ss=lgn_ss[i],
              rc=lgn_rc[i],
              rs=lgn_rs[i])]
        return tf.concat(axis=1, values=output)
    
    def cond(self, i, x, y, sc, ss, rc, rs):
      return i < self.num_lgn[0]  

    def build(self, data, label):
      get_bounds_for_init = lambda lo, up:  [lo + (up-lo)/4.0 , lo + (up-lo)/4.0 + (up-lo)/2.0]
      self.img_vec_size = int(data.get_shape()[-1])
      self.img_size = np.sqrt(self.img_vec_size)
      self.num_neurons = [int(label.get_shape()[-1])]

      grid_xx, grid_yy = tf.meshgrid(tf.range(self.img_size),tf.range(self.img_size))
      self.grid_xx = tf.cast(tf.reshape(grid_xx, [self.img_vec_size]), tf.float32)
      self.grid_yy = tf.cast(tf.reshape(grid_yy, [self.img_vec_size]), tf.float32)
      [L,U] = get_bounds_for_init(0, self.img_size)
      self.LGN_init = tf.random_uniform_initializer(L,U) #check here may be add another function?
      [L,U]=get_bounds_for_init(0.1, self.img_size)
      self.LGN_sc_init = tf.random_uniform_initializer(L,U)
      #import ipdb; ipdb.set_trace()
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
      checklowerbounds = lambda val, lower_bound : tf.maximum(val,lower_bound)
      self.hl_tresh = checklowerbounds(self.hl_tresh,0.0)
      self.ol_tresh= checklowerbounds(self.ol_tresh,0.0)

      self.l1 = self.activation(tf.matmul(self.lgn_out, self.hidden_w), self.hl_tresh) #RELU that shift
      self.output = self.activation(tf.matmul(self.l1, self.output_w), self.ol_tresh)
      
      self.LGN_params={'x_pos':self.lgn_x, 'y_pos':self.lgn_y, 'lgn_sc':self.lgn_sc, 'lgn_ss':self.lgn_ss, 'lgn_rc':self.lgn_rc, 'lgn_rs':self.lgn_rs}
      
      
      return self.output, self.l1, self.lgn_out
      def bounds(self):
        return self.bounds
        
      def der_all(self):
        return tf.grad(self.output, tf.trainable_variables())
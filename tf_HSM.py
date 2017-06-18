import numpy as np
import tensorflow as tf
import param


def logistic_loss(x,t=0, coef=1):
  return coef * tf.log(1 + tf.exp(coef*(x-t)))
  
class tf_HSM():
    '''
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
    '''

    def __init__(self, **params): #def __init__(**params):
        self.num_lgn=[9]
        self.hlsr = [0.2] 
        self.LGN_init = tf.constant_initializer(0) ############ Note the bound issue   
        self.LGN_sc_init = tf.constant_initializer(0.1)
        self.MLP_init = None
        self.activation = lambda x, y: logistic_loss(x, t=y, coef=1)
        self.images = None
        self.neural_response = None


    def construct_free_params(self,TrainHPY_PRM = False):
      # #LGN
      # self.lgn_x = tf.get_variable(name="x_pos", shape=self.num_lgn, initializer=self.LGN_init, trainable=False)
      # self.lgn_y = tf.get_variable(name="y_pos", shape=self.num_lgn, initializer=self.LGN_init, trainable=False)
      # self.lgn_sc = tf.get_variable(name="size_center", shape=self.num_lgn, initializer=self.LGN_sc_init, trainable=False) 
      # self.lgn_ss = tf.get_variable(name="size_surround", shape=self.num_lgn, initializer=self.LGN_init, trainable=False) 
      # self.lgn_rc = tf.get_variable(name="center_weight", shape=self.num_lgn, initializer=self.LGN_init, trainable=False) 
      # self.lgn_rs = tf.get_variable(name="surround_weight", shape=self.num_lgn, initializer=self.LGN_init, trainable=False)

      #LGN
      self.lgn_x = tf.get_variable(name="x_pos", shape=self.num_lgn, initializer=self.LGN_init, trainable=False)
      self.lgn_y = tf.get_variable(name="y_pos", shape=self.num_lgn, initializer=self.LGN_init, trainable=False)
      self.lgn_sc = tf.get_variable(name="size_center", shape=self.num_lgn, initializer=self.LGN_sc_init, trainable=False) 
      self.lgn_ss = tf.get_variable(name="size_surround", shape=self.num_lgn, initializer=self.LGN_init, trainable=False) 
      self.lgn_rc = tf.get_variable(name="center_weight", shape=self.num_lgn, initializer=self.LGN_init, trainable=False) 
      self.lgn_rs = tf.get_variable(name="surround_weight", shape=self.num_lgn, initializer=self.LGN_init, trainable=False) 

      # MLP
      self.hidden_w = tf.get_variable(
        name="hidden_weights",
        shape=(self.num_lgn[0], int(self.num_neurons[0]*self.hlsr[0])),
        initializer=self.LGN_init)  #init_bounds

      self.hl_tresh = tf.get_variable(
        name="hidden_layer_threshold",
        shape=int(self.num_neurons[0]*self.hlsr[0]),
        initializer=self.LGN_init) #init_bounds

      self.output_w = tf.get_variable(
        name="output_w",
        shape=(int(self.num_neurons[0]*self.hlsr[0]), int(self.num_neurons[0])),
        initializer=self.LGN_init) 

      self.ol_tresh = tf.get_variable(
        name="output_layer_threshold", #output_layer_threshold
        shape=int(self.num_neurons[0]),
        initializer=self.LGN_init) 
    
    def DoG(self, x, y, sc, ss, rc, rs):
      # Passing the parameters for a LGN neuron
      import ipdb; ipdb.set_trace()
      #x=14.62563043; y=19.43198948; sc=0.1; ss=1.85884457; rc= 0.45414222; rs=9.79140981;
      pos = ((self.grid_xx - x)**2 + (self.grid_yy - y)**2)
      center = tf.exp(-pos/2/sc) / (2*(sc)*np.pi)
      surround = tf.exp(-pos/2/(sc + ss)) / (2*(sc + ss)*np.pi)
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
        return tf.concat(1, output)
    
    def cond(self, i, x, y, sc, ss, rc, rs):
      return i < self.num_lgn[0]  

    def build(self, data, label):
      import ipdb; ipdb.set_trace()
      self.img_vec_size = int(data.get_shape()[-1])
      self.img_size = np.sqrt(self.img_vec_size)
      self.num_neurons = [int(label.get_shape()[-1])]

      grid_xx, grid_yy = tf.meshgrid(tf.range(self.img_size),tf.range(self.img_size))
      self.grid_xx = tf.cast(tf.reshape(grid_xx, [self.img_vec_size]), tf.float32)
      self.grid_yy = tf.cast(tf.reshape(grid_yy, [self.img_vec_size]), tf.float32)

      self.construct_free_params()

      self.images = data
      self.neural_response = label
      assert self.images is not None
      assert self.neural_response is not None
      
      # DoG
      self.lgn_out = self.LGN_loop(x_pos=self.lgn_x, y_pos=self.lgn_y, lgn_sc=self.lgn_sc, lgn_ss=self.lgn_ss, lgn_rc=self.lgn_rc, lgn_rs=self.lgn_rs)
      
      # Run MLP
      self.l1 = self.activation(tf.matmul(self.lgn_out, self.hidden_w), self.hl_tresh) #RELU that shict
      self.output = self.activation(tf.matmul(self.l1, self.output_w), self.ol_tresh)
      
      self.LGN_params={'x_pos':self.lgn_x, 'y_pos':self.lgn_y, 'lgn_sc':self.lgn_sc, 'lgn_ss':self.lgn_ss, 'lgn_rc':self.lgn_rc, 'lgn_rs':self.lgn_rs}
      
      
      return self.output, self.l1, self.lgn_out, self.LGN_params
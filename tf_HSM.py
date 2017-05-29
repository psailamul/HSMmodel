import numpy as np
import tensorflow as tf
import param


def logistic_loss(x,t=0, coef=1):
  return coef * tf.log(1 + tf.exp(coef*(x-t)))


def pearson_score(predictions, input_y):
  return tf.contrib.metrics.streaming_pearson_correlation(self.predictions, self.input_y, name="pearson")

  
class tf_HSM():
    '''
    Model for fitting vision data to neural recordings.

    params:

    XX = images

    Model:

    XX -> LGN -> 2-layer MLP
    LGN: Any vision feature extractor


    Output:

    Predicted neural responses.
    '''

    def __init__(self, **params): #def __init__(**params):
        self.num_lgn=[9]
        self.hlsr = [0.2]
        self.num_neurons =[103]
        self.LGN_init = tf.constant_initializer(0) ############ Note the bound issue   
        self.LGN_sc_init = tf.constant_initializer(0.1)
        self.MLP_init = None
        self.activation = lambda x: logistic_loss(x, coef=1)
        self.images = None
        self.neural_response = None
        self.construct_free_params()


    def construct_free_params(self):
      #LGN
      self.lgn_x = tf.get_variable(name="x_pos", shape=self.num_lgn, initializer=self.LGN_init)
      self.lgn_y = tf.get_variable(name="y_pos", shape=self.num_lgn, initializer=self.LGN_init)
      self.lgn_sc = tf.get_variable(name="size_center", shape=self.num_lgn, initializer=self.LGN_sc_init) 
      self.lgn_ss = tf.get_variable(name="size_surround", shape=self.num_lgn, initializer=self.LGN_init) 
      self.lgn_rc = tf.get_variable(name="center_weight", shape=self.num_lgn, initializer=self.LGN_init) 
      self.lgn_rs = tf.get_variable(name="surround_weight", shape=self.num_lgn, initializer=self.LGN_init) 

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
        shape=int(self.num_neurons[0]*self.hlsr[0]),
        initializer=self.LGN_init) 

      self.ol_tresh = tf.get_variable(
        name="output_layer_threshold", #output_layer_threshold
        shape=int(self.num_neurons[0]*self.hlsr[0]),
        initializer=self.LGN_init) 
    
    def DoG(self, x, y, sc, ss, rc, rs):
      # Passing the ith element of each variable
      img_vec_size=int(self.X.get_shape()[1])
      img_size = int(np.sqrt(img_vec_size ))
      num_LGN= sc.shape()[0]

      grid_xx, grid_yy = tf.meshgrid(tf.range(img_size),tf.range(img_size))
      grid_xx = tf.reshape(grid_xx, [img_vec_size])
      grid_yy = tf.reshape(grid_yy, [img_vec_size])

      pos = ((grid_xx - x)**2 + (grid_yy - y)**2)
      center = tf.exp(-pos/2/sc) / (2*(sc)*numpy.pi)
      surround = tf.exp(-pos/2/(sc + ss)) / (2*(sc + ss)*numpy.pi)
      return tf.matmul(self.images, (rc*(center)) - (rs*(surround)), transpose=True)

    def LGN(self, idx, x_pos, y_pos, lgn_sc, lgn_ss, lgn_rc, lgn_rs):
      #import pdb; pdb.set_trace()
      i = tf.to_int32(idx)
      output = self.DoG(
          x=x_pos[i],
          y=y_pos[i],
          sc=lgn_sc[i],
          ss=lgn_ss[i],
          rc=lgn_rc[i],
          rs=lgn_rs[i])
      idx+=1 
      return idx, output


    def cond(self, i, x, y, sc, ss, rc, rs):
      return tf.to_int32(i) < self.num_lgn[0]  

    def build(self, x, y):
      self.images=x
      self.neural_response=y
      assert self.images is not None
      assert self.neural_response is not None
      #import pdb; pdb.set_trace()
      i = tf.constant(0)
      LGN_vars = [i, x, y, self.lgn_sc, self.lgn_ss, self.lgn_rc, self.lgn_rs]
      self.lgn_out = tf.while_loop(body=self.LGN, cond=self.cond, loop_vars=LGN_vars, back_prop=False)

      # Run MLP
      #self.l1 = self.activation((lgn_out * self.hidden_w) - self.hl_tresh)
      #self.output = self.activation((l1 * self.output_w) - self.ol_tresh)
      self.l1 = self.activation(lgn_out * self.hidden_w, self.hl_tresh)
      self.output = self.activation(l1 * self.output_w, self.ol_tresh)
      #create loss
      loss = tf.nn.log_poisson_loss(self.output, self.neural_response, compute_full_loss=False)

      return self.output, loss

      
          # def LGN(self, i, x, y, sc, ss, rc, rs):
      # import pdb; pdb.set_trace()
      # idx = tf.to_int32(i)
      # output = self.DoG(
          # x=x[i],
          # y=y[i],
          # sc=sc[i],
          # ss=ss[i],
          # rc=rc[i],
          # rs=rs[i])
      # i+=1 
      # return i, output
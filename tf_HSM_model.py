"""
Implementation of HSM model in Tensorflow
"""
import numpy as np
import tensorflow as tf

def logistic_loss(x,t=0, coef=1):
  return coef * tf.log(1 + tf.exp(coef*(x-t)))

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

  def __init__(**params):

    self.size = None
    self.num_lgn = None
    self.hlsr = None
    self.v1of = None
    
	num_lgn = param.Integer(default=9,bounds=(0,10000),doc="""Number of lgn units""")
	hlsr = param.Number(default=0.2,bounds=(0,1.0),doc="""The hidden layer size ratio""")
	v1of = param.String(default='LogisticLoss',doc="""Transfer function of 'V1' neurons""")
	LL = param.Boolean(default=True,doc="""Whether to use Log-Likelyhood. False will use MSE.""")
		
    
    
    
    
    self.LGN_init = tf.constant_initializer(0) ############ Note the bound issue   
    self.LGN_sc_init = tf.constant_initializer(0.1)
    self.MLP_init = None
    self.activation = lambda x: logistic_loss(x, coef=1)

    self.construct_free_params(self)


  def construct_free_params(self):

    # LGN
    self.lgn_x = self.get_variable(name="x_pos", shape=self.num_lgn, initializer=self.LGN_init)
    self.lgn_y = self.get_variable(name="y_pos", shape=self.num_lgn, initializer=self.LGN_init)
    self.lgn_sc = self.get_variable(name="size_center", shape=self.num_lgn, initializer=self.LGN_sc_init) ### 
    self.lgn_ss = self.get_variable(name="size_surround", shape=self.num_lgn, initializer=self.LGN_init) ####
    self.lgn_rc = self.get_variable(name="center_weight", shape=self.num_lgn, initializer=self.LGN_init) ####
    self.lgn_rs = self.get_variable(name="surround_weight", shape=self.num_lgn, initializer=self.LGN_init) ###
    
     
    #Theano 
#    self.lgn_y = self.add_free_param("y_pos",self.num_lgn,(0,self.size))
 #   self.lgn_sc = self.add_free_param("size_center",self.num_lgn,(0.1,self.size))
  #  self.lgn_ss = self.add_free_param("size_surround",self.num_lgn,(0.0,self.size))
   	#self.lgn_rc = self.add_free_param("center_weight",self.num_lgn,(0.0,10.0))
    #self.lgn_rs = self.add_free_param("surround_weight",self.num_lgn,(0.0,10.0))

    # MLP
    self.hidden_w = self.get_variable(
      name="hidden_weights",
      shape=self.(self.num_lgn,int(self.num_neurons*self.hlsr)),
      initializer=self.LGN_init())  #init_bounds
      
    self.hl_tresh = self.get_variable(
      name="hidden_layer_threshold",
      shape=int(self.num_neurons*self.hlsr),
      initializer=self.LGN_init()) #init_bounds
	
	self.output_w = self.get_variable(
      name="hidden_layer_threshold",
      shape=int(self.num_neurons*self.hlsr),
      initializer=self.LGN_init()) 
      
    self.ol_tresh = self.get_variable(
      name="hidden_layer_threshold",
      shape=int(self.num_neurons*self.hlsr),
      initializer=self.LGN_init()) 
      
      
    self.output_w = self.add_free_param("output_weights",(int(self.num_neurons*self.hlsr),self.num_neurons),(None,None),init_bounds=(-10,10))
    self.ol_tresh = self.add_free_param("output_layer_threshold",int(self.num_neurons),(0,None),init_bounds=(0,10))


  def LGN(self, i, x, y, sc, ss, rc, rs):
    return tf.dot(
      self.X,rc[i]*(T.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/sc[i]).T/ (2*sc[i]*numpy.pi))\
      - rs[i]*(T.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/(sc[i]+ss[i])).T/ (2*(sc[i]+ss[i])*numpy.pi)))

  def cond(self, i, x, y, sc, ss, rc, rs):
    if cond > self.num_lgn:
      return
    else:
      cond += 1
    return i, x, y, sc, ss, rc, rs

  def build(self, XX):
    # Run LGN
    LGN_vars = [x, y, self.lgn_sc, self.lgn_ss, self.lgn_rc, self.lgn_rs]
    self.lgn_out = tf.while_loop(body=LGN, cond=cond, loop_vars=LGN_vars, back_prop=False)

    # Run MLP
    self.l1 = self.activation((lgn_out * self.hidden_w) - self.hl_tresh)
    self.output = self.activation((l1 * self.output_w) - self.ol_tresh)

    return self.output
	
	
	
  """


      #Theano 
  #    self.lgn_y = self.add_free_param("y_pos",self.num_lgn,(0,self.size))
   #   self.lgn_sc = self.add_free_param("size_center",self.num_lgn,(0.1,self.size))
    #  self.lgn_ss = self.add_free_param("size_surround",self.num_lgn,(0.0,self.size))
      #self.lgn_rc = self.add_free_param("center_weight",self.num_lgn,(0.0,10.0))
      #self.lgn_rs = self.add_free_param("surround_weight",self.num_lgn,(0.0,10.0))

        #LGN
    # self.lgn_x = tf.get_variable(name="x_pos", shape=[9], initializer=self.LGN_init)
    # self.lgn_y = tf.get_variable(name="y_pos", shape=[9], initializer=self.LGN_init)
    # self.lgn_sc = tf.get_variable(name="size_center", shape=[9], initializer=self.LGN_sc_init) ### 
    # self.lgn_ss = tf.get_variable(name="size_surround", shape=[9], initializer=self.LGN_init) ####
    # self.lgn_rc = tf.get_variable(name="center_weight", shape=[9], initializer=self.LGN_init) ####
    # self.lgn_rs = tf.get_variable(name="surround_weight", shape=[9], initializer=self.LGN_init) ###
  """
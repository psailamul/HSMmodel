import numpy as np
import tensorflow as tf
import param
from tf_HSM import tf_HSM

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


  # class tf_HSM():
    # '''
    # Model for fitting vision data to neural recordings.

    # params:

    # XX = images

    # Model:

    # XX -> LGN -> 2-layer MLP
    # LGN: Any vision feature extractor


    # Output:

    # Predicted neural responses.
    # '''

    # def __init__(self, **params): #def __init__(**params):
      # self.num_lgn=[9]
      # self.hlsr = [0.2]
      # self.num_neurons =[103]
      # self.LGN_init = tf.constant_initializer(0) ############ Note the bound issue   
      # self.LGN_sc_init = tf.constant_initializer(0.1)
      # self.MLP_init = None
      # self.activation = lambda x: logistic_loss(x, coef=1)
      # self.images = None
      # self.neural_response = None
      # self.construct_free_params()


    # def construct_free_params(self):
      # #LGN
      # self.lgn_x = tf.get_variable(name="x_pos", shape=self.num_lgn, initializer=self.LGN_init)
      # self.lgn_y = tf.get_variable(name="y_pos", shape=self.num_lgn, initializer=self.LGN_init)
      # self.lgn_sc = tf.get_variable(name="size_center", shape=self.num_lgn, initializer=self.LGN_sc_init) 
      # self.lgn_ss = tf.get_variable(name="size_surround", shape=self.num_lgn, initializer=self.LGN_init) 
      # self.lgn_rc = tf.get_variable(name="center_weight", shape=self.num_lgn, initializer=self.LGN_init) 
      # self.lgn_rs = tf.get_variable(name="surround_weight", shape=self.num_lgn, initializer=self.LGN_init) 

      # # MLP
      # self.hidden_w = tf.get_variable(
        # name="hidden_weights",
        # shape=(self.num_lgn[0], int(self.num_neurons[0]*self.hlsr[0])),
        # initializer=self.LGN_init)  #init_bounds

      # self.hl_tresh = tf.get_variable(
        # name="hidden_layer_threshold",
        # shape=int(self.num_neurons[0]*self.hlsr[0]),
        # initializer=self.LGN_init) #init_bounds

      # self.output_w = tf.get_variable(
        # name="output_w",
        # shape=int(self.num_neurons[0]*self.hlsr[0]),
        # initializer=self.LGN_init) 

      # self.ol_tresh = tf.get_variable(
        # name="output_layer_threshold", #output_layer_threshold
        # shape=int(self.num_neurons[0]*self.hlsr[0]),
        # initializer=self.LGN_init) 
    
    # def DoG(self, x, y, sc, ss, rc, rs):
      # # Passing the ith element of each variable

      # img_vec_size=int(self.X.get_shape()[1])
      # img_size = int(np.sqrt(img_vec_size ))
      # num_LGN= sc.shape()[0]

      # grid_xx, grid_yy = tf.meshgrid(tf.range(img_size),tf.range(img_size))
      # grid_xx = tf.reshape(grid_xx, [img_vec_size])
      # grid_yy = tf.reshape(grid_yy, [img_vec_size])

      # pos = ((grid_xx - x)**2 + (grid_yy - y)**2)
      # center = tf.exp(-pos/2/sc) / (2*(sc)*numpy.pi)
      # surround = tf.exp(-pos/2/(sc + ss)) / (2*(sc + ss)*numpy.pi)
      # return tf.matmul(self.images, (rc*(center)) - (rs*(surround)), transpose=True)

    # def LGN(self, i, x, y, sc, ss, rc, rs):
      # output = DoG(
          # x=x[i],
          # y=y[i],
          # sc=sc[i],
          # ss=ss[i],
          # rc=rc[i],
          # rs=rs[i])
      # i+=1 
      # return i, output


    # def cond(self, i, x, y, sc, ss, rc, rs):
      # return i < self.num_lgn # check again  

    # def build(self, x, y):
      # # Run LGN
      # assert self.images is not None
      # assert self.neural_response is not None

      # i = tf.constant(0)
      # LGN_vars = [i, x, y, self.lgn_sc, self.lgn_ss, self.lgn_rc, self.lgn_rs]
      # self.lgn_out = tf.while_loop(body=self.LGN, cond=self.cond, loop_vars=LGN_vars, back_prop=False)

      # # Run MLP
      # self.l1 = self.activation((lgn_out * self.hidden_w) - self.hl_tresh)
      # self.output = self.activation((l1 * self.output_w) - self.ol_tresh)
	  # #create loss
      # loss = tf.nn.log_poisson_loss(self.output, self.neural_response, compute_full_loss=False)

      # return self.output, loss




# Main script
dt_stamp = '17_05_29'

images = tf.placeholder(tf.float32)
neural_response = tf.placeholder(tf.float32)
lr = 1e-3
iterations = 100

# Download data from a region
train_input=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_inputs.npy')
train_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_set.npy')
#load trained LGN hyperparameters
#num_LGN = NUM_LGN; hlsr = HLSR;
[Ks,hsm] = np.load('out_region1.npy')

#load trained parameters for DoG
#x,y,sc,ss,rc,rs = get_trained_Ks(Ks,9)
#import pdb; pdb.set_trace()

with tf.device('/gpu:0'):
  with tf.variable_scope('hsm') as scope:
	# import pdb; pdb.set_trace()

    # Declare and build model
    hsm = tf_HSM()
    model, loss = hsm.build(images, neural_response)

    # Define loss
    #loss = tf.nn.l2_loss(model.output, YY)

    # Optimize loss
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    # Track correlation between YY_hat and YY
    score = pearson_score(model.output, YY)

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
summary_dir = os.path.join(  "TFtrainingSummary/"  'AntolikRegion2_' + dt_stamp)# pachaya declare a directory to store summaries here!
   # config.train_summaries, config.which_dataset + '_' + dt_stamp)
summary_writer = tf.train.SummaryWriter(summary_dir, sess.graph)




for idx in range(iterations):
	_, loss_value, correlation = sess.run(
	  [train_op, loss, score],
	  feed_dict={'images': train_input, 'neural_response':train_set})


####################################### 

"""
import numpy as np
import tensorflow as tf
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
  tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
  mode=mode, predictions=y,
  loss=loss,
  train_op=train)

  estimator = tf.contrib.learn.Estimator(model_fn=model)
  # define our data set
  x = np.array([1., 2., 3., 4.])
  y = np.array([0., -1., -2., -3.])
  input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

  # train
  estimator.fit(input_fn=input_fn, steps=1000)
  # evaluate our model
  print(estimator.evaluate(input_fn=input_fn, steps=10))
  """


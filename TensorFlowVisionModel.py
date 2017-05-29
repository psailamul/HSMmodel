import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def logistic_loss(x,t=0, coef=1):
  return coef * tf.log(1 + tf.exp(coef*(x-t)))
  
  
def log_loss2(x,t=0, coef=1):
  return coef * np.log(1 + np.exp(coef*(x-t)))
  
  
 def computeCorr(pred_act,responses):
    """
    Compute correlation between predicted and recorded activity for each cell
    """

    num_pres,num_neurons = np.shape(responses)
    corr=np.zeros(num_neurons)
    
    for i in xrange(0,num_neurons):
        if np.all(pred_act[:,i]==0) & np.all(responses[:,i]==0):
            corr[i]=1.
        elif not(np.all(pred_act[:,i]==0) | np.all(responses[:,i]==0)):
            # /!\ To prevent errors due to very low values during computation of correlation
            if abs(pred_act[:,i]).max()<1:
                pred_act[:,i]=pred_act[:,i]/abs(pred_act[:,i]).max()
            if abs(responses[:,i]).max()<1:
                responses[:,i]=responses[:,i]/abs(responses[:,i]).max()    
            corr[i]=pearsonr(np.array(responses)[:,i].flatten(),np.array(pred_act)[:,i].flatten())[0]
            
    return corr 
  
  
def model(features, labels, mode):
# ###################################### From Tutorial
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
	
	
	
	
	
	# ###################################### From Tutorial
	
	
	return tf.contrib.learn.ModelFnOps(
	  mode=mode, predictions=y,
	  loss=loss,
	  train_op=train)

  
 ####################################### From Drew
 
 
 
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

def cmpr_hist_pred_record(P, Y, num_bins = 20):
	
	input_shape = np.shape(P)
	PP = np.reshape(P,[input_shape[0]*input_shape[1],1])
	YY = np.reshape(Y,[input_shape[0]*input_shape[1],1])
	
	# Two subplots, unpack the axes array immediately
	f, (ax1, ax2) = plt.subplots(1, 2)
	ax1.hist(YY,num_bins)
	ax1.set_title('Recorded neural response')
	ax2.hist(PP,num_bins)
	ax2.set_title('Predicted response')
	plt.show()

 
 
 ###################################
  def __init__(**params):
	"""
    self.size = None
    self.num_lgn = None
    self.hlsr = None
    self.v1of = None
    """
    
    self.num_lgn = param.Integer(default=9,bounds=(0,10000),doc="""Number of lgn units""")
    self.hlsr = param.Number(default=0.2,bounds=(0,1.0),doc="""The hidden layer size ratio""")
    self.v1of = param.String(default='LogisticLoss',doc="""Transfer function of 'V1' neurons""")
    self.LL = param.Boolean(default=True,doc="""Whether to use Log-Likelyhood. False will use MSE.""")
		
    
    
    
    
    self.LGN_init = tf.constant_initializer(0) ############ Note the bound issue   
    self.LGN_sc_init = tf.constant_initializer(0.1)
    self.MLP_init = None
    self.activation = lambda x: logistic_loss(x, coef=1)

    self.construct_free_params(self)

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
##############################################################	
# Main script
XX = tf.placeholder(shape=None)
YY = tf.placeholder(shape=None)
iterations = 100
lr = 1e-3 #learning rate

NUM_LGN = 9; HLSR = 0.2 

train_input=np.load('/home/pachaya/AntolikData/SourceCode/Data/region2/training_inputs.npy')
train_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region2/training_set.npy')

#load trained LGN hyperparameters
num_LGN = NUM_LGN; hlsr = HLSR;
[Ks,hsm] = np.load('out_region2.npy')

#load trained parameters for DoG
x,y,sc,ss,rc,rs = get_trained_Ks(Ks,9)

# num LGN = 9 L1 = 20  L2 = 103
# per image
X=np.ndarray([1800,9]) #LGN output = input to NN
for ii in np.arange(np.shape(train_input)[0]):
	lgn_ker_out = get_LGN_out(train_input[ii,:],x,y,sc,ss,rc,rs) # Output of LGN = input to NN
	X[ii,:]=lgn_ker_out #Output from LGN
# X = features
Y = train_set.copy() #1 All image = 1800,103

input_fn = tf.contrib.learn.io.numpy_input_fn({"x":X}, Y, batch_size=4,
                                              num_epochs=500)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
print(estimator.evaluate(input_fn=input_fn))


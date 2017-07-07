import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re

def logistic_loss(x,t=0, coef=1):
  return coef * np.log(1 + np.exp(coef*(x-t)))
  
class test_model():
    def __init__(self, **params): #def __init__(**params):
        self.num_lgn=[9]
        self.hlsr = [0.2] 
        self.LGN_init =0
        self.LGN_sc_init = 0.1
        self.MLP_init = None
        self.activation = lambda x, y: logistic_loss(x, t=y, coef=1)
        self.images = None
        self.neural_response = None
        
    def DoG(self, x, y, sc, ss, rc, rs):
        # Passing the parameters for a LGN neuron

        pos = ((self.grid_xx - x)**2 + (self.grid_yy - y)**2)
        center = np.exp(-pos/2/sc) / (2*(sc)*np.pi)
        surround = np.exp(-pos/2/(sc + ss)) / (2*(sc + ss)*np.pi)
        weight_vec = np.reshape((rc*(center)) - (rs*(surround)), [-1, 1])
        return np.matmul(self.images, weight_vec)

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
        return np.concatenate(1, output)

    def build(self, data, label):

        self.img_vec_size = int(data.get_shape()[-1])
        self.img_size = np.sqrt(self.img_vec_size)
        self.num_neurons = [int(label.get_shape()[-1])]

        grid_xx, grid_yy = np.meshgrid(np.range(self.img_size),np.range(self.img_size))
        self.grid_xx = np.cast(np.reshape(grid_xx, [self.img_vec_size]), np.float32)
        self.grid_yy = np.cast(np.reshape(grid_yy, [self.img_vec_size]), np.float32)

        self.construct_free_params()

        self.images = data
        self.neural_response = label
        assert self.images is not None
        assert self.neural_response is not None
        # DoG
        self.lgn_out = self.LGN_loop(x_pos=self.lgn_x, y_pos=self.lgn_y, lgn_sc=self.lgn_sc, lgn_ss=self.lgn_ss, lgn_rc=self.lgn_rc, lgn_rs=self.lgn_rs)

        # Run MLP
        self.l1 = self.activation(np.matmul(self.lgn_out, self.hidden_w), self.hl_tresh) #RELU that shict
        self.output = self.activation(np.matmul(self.l1, self.output_w), self.ol_tresh)

        return self.output, self.lgn_out
        
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

    lgn_ker_out = np.ndarray([num_LGN])

    for i in np.arange(num_LGN):
        lgn_ker_out[i] = lgn_kernel(i,x,y,sc,ss,rc,rs)

    return lgn_ker_out

# Main script
########################################################################
dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
        
region_num = '1'
########################################################################
import ipdb; ipdb.set_trace()
# Download data from a region
train_input=np.load('/home/pachaya/AntolikData/SourceCode/Data/region' + region_num+'/training_inputs.npy')
train_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region' + region_num+'/training_set.npy')

#hsm = test_model()
#pred_neural_response,lgn_out = hsm.build(images, neural_response)



#load data
train_input=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_inputs.npy') # 1800, 961
train_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_set.npy') #1800, 103
#load validation data
raw_vld_set = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/raw_validation_set.npy')
vldinput_set = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/validation_inputs.npy')
vld_set = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/validation_set.npy')

#size 
img_vec_size =int(np.shape(train_input)[1])
img_size = int(np.sqrt(img_vec_size ))


#load trained LGN hyperparameters
NUM_LGN = 9; HLSR =0.2;
num_LGN = NUM_LGN; hlsr = HLSR;
[Ks,hsm] = np.load('out_region1.npy')

#load trained parameters for DoG
x,y,sc,ss,rc,rs = get_trained_Ks(Ks,9)

# num LGN = 9 L1 = 20  L2 = 103
# per image
X=np.ndarray([1800,9]) #LGN output = input to NN
for ii in np.arange(np.shape(train_input)[0]):
	lgn_ker_out = get_LGN_out(train_input[ii,:],x,y,sc,ss,rc,rs) # Output of LGN = input to NN
	X[ii,:]=lgn_ker_out #Output from LGN

Y = train_set.copy() #1 All image = 1800,103


images=train_input[0,:]
pos = ((grid_xx - x)**2 + (grid_yy - y)**2)
center = np.exp(-pos/2/sc) / (2*(sc)*np.pi)
surround = np.exp(-pos/2/(sc + ss)) / (2*(sc + ss)*np.pi)
weight_vec = np.reshape((rc*(center)) - (rs*(surround)), [-1, 1])
llggnn = np.matmul(images, weight_vec)
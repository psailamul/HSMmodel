"""
This file contains the implementation of the HSM model.
"""

import numpy
import theano
import param

from theano import tensor as T
from TheanoVisionModel import TheanoVisionModel

class HSM(TheanoVisionModel):
      
      num_lgn = param.Integer(default=9,bounds=(0,10000),doc="""Number of lgn units""")
      hlsr = param.Number(default=0.2,bounds=(0,1.0),doc="""The hidden layer size ratio""")
      v1of = param.String(default='LogisticLoss',doc="""Transfer function of 'V1' neurons""")
      LL = param.Boolean(default=True,doc="""Whether to use Log-Likelyhood. False will use MSE.""")

      def construct_free_params(self):

            # NOTE: def add_free_param(self,name,shape,bounds,init_bounds=None):
            #       return a TensorVariable datatype
            # LGN
            self.lgn_x = self.add_free_param("x_pos",self.num_lgn,(0,self.size)) # shape = num_lgn = 9 , bound = (0,31)
            self.lgn_y = self.add_free_param("y_pos",self.num_lgn,(0,self.size))
            self.lgn_sc = self.add_free_param("size_center",self.num_lgn,(0.1,self.size))  # shape = num_lgn = 9 , bound = (0.1,31)
            self.lgn_ss = self.add_free_param("size_surround",self.num_lgn,(0.0,self.size))

            self.lgn_rc = self.add_free_param("center_weight",self.num_lgn,(0.0,10.0)) # weight = 0 - 10
            self.lgn_rs = self.add_free_param("surround_weight",self.num_lgn,(0.0,10.0))


            self.hidden_w = self.add_free_param("hidden_weights",(self.num_lgn,int(self.num_neurons*self.hlsr)),(None,None),init_bounds=(-10,10)) #shape = (9, int(103*0.2)) = (9, 20)   , init_bounds=(-10,10)
            self.hl_tresh = self.add_free_param("hidden_layer_threshold",int(self.num_neurons*self.hlsr),(0,None),init_bounds=(0,10))  #shape = int(103*0.2) = 20   , init_bounds=(0,10) 
            self.output_w = self.add_free_param("output_weights",(int(self.num_neurons*self.hlsr),self.num_neurons),(None,None),init_bounds=(-10,10)) #shape = (int(103*0.2),103) = (20, 103)   , init_bounds=(-10,10) 
            self.ol_tresh = self.add_free_param("output_layer_threshold",int(self.num_neurons),(0,None),init_bounds=(0,10)) #shape = 103, init_bounds=(0,10)



      def construct_model(self):
            # construct the 'retinal' x and y coordinates matrices
            xx = theano.shared(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).T.flatten()) # similar to mesh grid  31x31 matrix, 0-30 in each row
            yy = theano.shared(numpy.repeat([numpy.arange(0,self.size,1)],self.size,axis=0).flatten()) # similar to mesh grid  31x31 matrix, 0-30 in each row
            # numpy.repeat([np.arange(0,hsm.size,1)],hsm.size,axis=0)
            
            # Initialize your DoG
            lgn_kernel = lambda i,x,y,sc,ss,rc,rs: T.dot(
                self.X,rc[i]*(T.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/sc[i]).T/ (2*sc[i]*numpy.pi))\
                - rs[i]*(T.exp(-((xx - x[i])**2 + (yy - y[i])**2)/2/(sc[i]+ss[i])).T/ (2*(sc[i]+ss[i])*numpy.pi))) # X = input image 
            # Apply the DoG to the Data
            lgn_output, updates = theano.scan(  # tf.while_loop() ######################################################################### Check here again, if it's the same as tf.while_loop()
                lgn_kernel,
                sequences=T.arange(self.num_lgn),
                non_sequences=[self.lgn_x,self.lgn_y,self.lgn_sc,self.lgn_ss,self.lgn_rc,self.lgn_rs])
            
            # Apply non-linearity to DoG output
            lgn_output = lgn_output.T
            lgn_output = self.construct_of(lgn_output, 'Linear')

            # Combine LGN with (right now random) weights: L1
            output = T.dot(lgn_output,self.hidden_w)
            model_output = self.construct_of(output-self.hl_tresh,self.v1of) #output from hidden layer
            #
            # Me 
            #model_hl_output = self.construct_of(output-self.hl_tresh,self.v1of)
             
            #presynapses_hl_v1 = T.dot(model_hl_output , self.output_w) # input to L2 = L1 response * W_L1_to_L2
            #model_L2_output = self.construct_of(presynapses_hl_v1-self.hl_tresh,self.v1of) 
             
            # L2
            model_output = self.construct_of(T.dot(model_output , self.output_w) - self.ol_tresh,self.v1of) #implement threshold outside function

            self.model_output = model_output

            return model_output

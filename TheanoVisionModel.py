import numpy
import param
import theano
theano.config.floatX='float32'
from theano import tensor as T
from theano import function, config, shared
from numpy.random import rand, seed

class TheanoVisionModel(param.Parameterized):
        """
        This class offers helper functionality for constructing models from Theano constructs.

        The two inputs to the constructor are:

        XX - the training inputs matrix
        YY - the training outputs matrix

        User is supposed to define two main functions in derived classes:

        construct_free_params()
        construct_model()

        See their documentation for their role.
        """

        error_function = param.String(default="LogLikelyhood", doc="The error function definition. Currently LogLikelyhood and MSE are accapted options.")
        log_loss_coefficient = param.Number(default=1.0, doc="The coeficcient that controls the sharpness of the log-loss function elbow.")

        def __init__(self,XX,YY,**params):
          """
          The constructor sets up various Theano elements that we will need.
          Once that is done it constructs the model, by calling in sequence the
          construct_free_params, construct_model and construct_error_function
          functions, first two of which should be overriden by the specific
          model.
          """
          param.Parameterized.__init__(self,**params)

          assert numpy.shape(XX)[0] == numpy.shape(YY)[0]
          (self.num_pres,self.kernel_size) = numpy.shape(XX)
          (self.num_pres,self.num_neurons) = numpy.shape(YY)


          self.X = theano.shared(XX) # the training inputs matrix
          self.Y = theano.shared(YY) # the training ouputs matrix

          self.size = numpy.sqrt(self.kernel_size)

          self.free_params = {}
          self.free_param_count = 0
          # K as a variable in expression
          self.K = T.dvector('K') #This will hold the free-parameters Theano vector ************************************************
          self.bounds = [] #This will hold the bounds
          self.init_bounds = [] #This will hold the bounds for parameter initialization

          self.construct_free_params()
          output = self.construct_model() #**************************************************************************************
          self.construct_error_function(output)


        def add_free_param(self,name,shape,bounds,init_bounds=None):
          """
          Add a free param with name <name>, with shape <shape> and with bounds <bounds>.
          Shape should be either a number (when parameter is vector) or tuple (when parameter is matrix).
          Bounds should be tuple (min,max)
          """
          if type(shape) == tuple:
             self.free_params[name] = (self.free_param_count,shape) # --> tuple that tell start index and shape of each "type" of parameter ex. 'center_weight': (36, 9)
             self.free_param_count = self.free_param_count + shape[0]*shape[1]
             for i in xrange(0,shape[0]*shape[1]):
                 self.bounds.append(bounds) # self.bounds = list of bounds for each parameter (the exact parameter) --> size = [free_param_count , 2] ex [2435, 2]
                 if init_bounds != None:
                    self.init_bounds.append(init_bounds) #bounds when initialize the parameter
                 else:
                    self.init_bounds.append(bounds) # If the initialize bounds doesn't specify --> set initialized bound to the overall bound
          else: #shape size = scalar ex. self.num_lgn (=9)
             self.free_params[name] = (self.free_param_count,shape)
             self.free_param_count = self.free_param_count + shape
             for i in xrange(0,shape):
                 self.bounds.append(bounds)
                 if init_bounds != None:
                    self.init_bounds.append(init_bounds)
                 else:
                    self.init_bounds.append(bounds)

          return self.getParam(self.K,name)

        def construct_free_params(self):
          """
          This function should hold calls to add_free_param that will construct
          data structures that will handle input to the model.
          """
          raise NotImplementedError()
          pass


        def construct_model(self):
          """
          This function should construct the model. It can reffer to the params constructed
          in construct_free_params() by their names using the function gfp.

          The construct_model has to return the model
          (essentially it returns Theano variable that holds the
          vector with the output neurons in the output layer - i.e.
          correspodning to the recorded neurons)
          """
          raise NotImplementedError()
          pass


        def construct_error_function(self,model_output):
            """
            This function specifies which of the error (or cost) functions will be used.
            """
            if self.error_function == 'LogLikelyhood':
               self.model = T.sum(model_output) - T.sum(self.Y * T.log(model_output+0.0000000000000000001))
            elif self.error_function == 'MSE':
               self.model = T.sum(T.sqr(model_output - self.Y))

        def func(self):
            """
            Returns theano created function, that takes model parametrization as input (a vector of floats), and returns the
            response of the model to the training set stored in self.X as output.
            """
            #https://theano.readthedocs.io/en/rel-0.6rc3/library/compile/function.html#
            return theano.function(inputs=[self.K], outputs=self.model,mode='FAST_RUN') #return (scalar) loss
            

        def der(self):
            """
            Returns theano created function, that takes model parametrization as input (a vector of floats), and returns the
            response of the first derivative of the model to the training set stored in self.X as output.
            """
            g_K = T.grad(self.model, self.K) #T.grad(scalar cost, variable to compute gradient)
            return theano.function(inputs=[self.K], outputs=g_K,mode='FAST_RUN')

        def response(self,X,kernel,mode='FAST_RUN'):
            """
            This function takes as input some model inputs, and a parametrization of the model, and returns the
            response of the model, with respect to the parametrization and inputs.
            """
            self.X.set_value(X)
            resp = theano.function(inputs=[self.K], outputs=self.model_output,mode=mode)
            return resp(kernel)

        def construct_of(self,inn,of):
            """
            A function providing implementation of commonly used output functions.
            """
            if of == 'Linear':
               return inn
            if of == 'Exp':
               return T.exp(inn)
            elif of == 'Sigmoid':
               return 5.0 / (1.0 + T.exp(-inn))
            elif of == 'SoftSign':
               return inn / (1 + T.abs_(inn))
            elif of == 'Square':
               return T.sqr(inn)
            elif of == 'ExpExp':
               return T.exp(T.exp(inn))
            elif of == 'ExpSquare':
               return T.exp(T.sqr(inn))
            elif of == 'LogisticLoss':
               return self.log_loss_coefficient*T.log(1+T.exp(self.log_loss_coefficient*inn))

        def getParam(self,param_vector,param_name):
            """
            #param_vector = self.K 
            Returns the subvector of param_vector corresponding to the parameter in param_name.
            Note that this function returns a TensorVariable datatype.
            """
            if type(self.free_params[param_name][1]) == tuple:
                    (i,t) = self.free_params[param_name]
                    (x,y) = t
                    return T.reshape(param_vector[i:i+x*y],(x,y))
            else:
                    (i,l) = self.free_params[param_name]
                    return param_vector[i:i+l]


        def printParams(self,param_vector,param_names=None):
            """
            Prints the name and values for each model parameter for the parametrization vector param_vector.
            """
            if param_names == None:
               param_names = self.free_params.keys()

            print "Model parameters:"
            for p in param_names:
                if type(self.free_params[p][1]) == tuple:
                        (i,(x,y)) = self.free_params[p]
                        print p, ": ", numpy.reshape(param_vector[i:i+x*y],(x,y))
                else:
                        (i,l) = self.free_params[p]
                        print i
                        print p, ": ", param_vector[i:i+l]

        def create_random_parametrization(self,s):
            """
            This function creates are random parametrization of the model. All values of the returned
            parameter vector are taken from uniform distribution bounded by the bounds associated
            with each parameter.

            The seed parameter sets the seed of the random number generator used to draw the random vector.
            """
            seed(s)
            return [a[0] + (a[1]-a[0])/4.0 + rand()*(a[1]-a[0])/2.0  for a in self.init_bounds] #Distribution of rand() = uniform

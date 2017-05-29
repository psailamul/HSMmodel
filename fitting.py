from HSM import HSM
import numpy
from scipy.optimize import fmin_tnc
import param

def fitHSM(training_inputs,training_set,seed=13,lgn=9,hlsr=0.2):
    """
    This function performs fitting of the HSM model using the fmin_tnc numpy method.

    training_inputs : 2D ndarray of inputs of shape (num of training presentations,number of pixels)
    training_set    : 2D ndarray of neural responses to corresponding inputs of shape (num of training presentations,number of recorded neurons)
    """
    num_pres,num_neurons = numpy.shape(training_set)
    #import ipdb; ipdb.set_trace()
    print "Creating HSM model"
    hsm = HSM(training_inputs,training_set)
    print "Created HSM model"   
    hsm.num_lgn = lgn 
    hsm.hlsr = hlsr
    # create the theano loss function
    func = hsm.func()
    #import ipdb; ipdb.set_trace()
    # set initial random values of the model parameter vector
    Ks = hsm.create_random_parametrization(seed)

    (Ks,success,c)=fmin_tnc(func ,Ks,fprime=hsm.der(),bounds=hsm.bounds,maxfun = 100000,messages=0)  # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_tnc.html
    print 'Final training error: ', func(numpy.array(Ks))/num_neurons/len(training_set)

    return [Ks,hsm]

    

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
    import ipdb; ipdb.set_trace()
    print "Creating HSM model"
    hsm = HSM(training_inputs,training_set) # Initialize model --> add input and output, construct parameters , build mobel, # create loss function
    print "Created HSM model"   
    hsm.num_lgn = lgn 
    hsm.hlsr = hlsr
    
    func = hsm.func() 
    #import ipdb; ipdb.set_trace()
    Ks = hsm.create_random_parametrization(seed) # set initial random values of the model parameter vector

    #(Ks,success,c)=fmin_tnc(func ,Ks,fprime=hsm.der(),bounds=hsm.bounsd,maxfun = 100000,messages=0)  # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_tnc.html
    (Ks,success,c)=fmin_tnc(func ,Ks,fprime=hsm.der(),bounds=hsm.bounds,maxfun = 100000,disp=5)
    """ 
    	
	func : callable func(x, *args)
		Function to minimize. Must do one of:
			Return f and g, where f is the value of the function and g its gradient (a list of floats).
			Return the function value but supply gradient function separately as fprime.
			Return the function value and set approx_grad=True.
		If the function returns None, the minimization is aborted.
	x0 : array_like
		Initial estimate of minimum.
	fprime : callable fprime(x, *args)
		Gradient of func. If None, then either func must return the function value and the gradient (f,g = func(x, *args)) or approx_grad must be True
    bounds : list
		(min, max) pairs for each element in x0, defining the bounds on that parameter. Use None or +/-inf for one of min or max when there is no bound in that direction.

    maxfun : int
		Maximum number of function evaluation. if None, maxfun is set to max(100, 10*len(x0)). Defaults to None.
	
	messages :
		Bit mask used to select messages display during minimization values defined in the MSGS dict. Defaults to MGS_ALL.
		
	Returns:	
		x : ndarray
			The solution.
		nfeval : int
			The number of function evaluations.
		rc : int
			Return code as defined in the RCSTRINGS dict.

Minimize a function with variables subject to bounds, 
using gradient information in a truncated Newton algorithm. 
This method wraps a C implementation of the algorithm.
	"""
    print 'Final training error: ', func(numpy.array(Ks))/num_neurons/len(training_set)

    return [Ks,hsm]

    

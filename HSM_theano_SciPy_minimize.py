from HSM import HSM
import numpy
from scipy.optimize import minimize
import param

import numpy as np
from fitting import fitHSM
import time
from get_host_path import get_host_path

download_time = time.time()

training_inputs=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_inputs.npy')
training_set=np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/training_set.npy')
print "Download complete: Time %s" %(time.time() - download_time)
call_time = time.time()
seed=13; lgn=9; hlsr=0.2
import ipdb; ipdb.set_trace()
num_pres,num_neurons = numpy.shape(training_set)
print "Creating HSM model"
hsm = HSM(training_inputs,training_set) # Initialize model --> add input and output, construct parameters , build mobel, # create loss function
print "Created HSM model"   
hsm.num_lgn = lgn 
hsm.hlsr = hlsr
    
func = hsm.func() 
import ipdb; ipdb.set_trace()
Ks = hsm.create_random_parametrization(seed) # set initial random values of the model parameter vector

MAXITER=100000
Code='SciPytestSeed'
HOST, PATH = get_host_path(HOST=True, PATH=True)
#SUMMARY_DIR = 'TFtrainingSummary/SciPy_maxiter_grad/'
SUMMARY_DIR = 'TFtrainingSummary/SciPy_SEEDnumpy/'
#c(Ks,success,c)=fmin_tnc(func ,Ks,fprime=hsm.der(),bounds=hsm.bounsd,maxfun = 100000,messages=0)  # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_tnc.html
out=minimize(func ,Ks,method='TNC',jac=hsm.der(),bounds=hsm.bounds,options={'maxiter':MAXITER,'disp':True})
import ipdb; ipdb.set_trace()
Ks=out.x
np.save("%sHSMout_theano_%s_MaxIter%g_seed%g.npy"%(SUMMARY_DIR,Code,MAXITER,seed),out)
print "Saved"

print 'Final training error: ', func(numpy.array(Ks))/num_neurons/len(training_set)
runtime = time.time() - call_time

import ipdb; ipdb.set_trace()
print "Finish training %s" %(runtime)



raw_vld_set = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/raw_validation_set.npy')
vldinput_set = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/validation_inputs.npy')
vld_set = np.load('/home/pachaya/AntolikData/SourceCode/Data/region1/validation_set.npy')

response = HSM.response(hsm,training_inputs,Ks) # predicted response after train

#Validation set
pred_response = HSM.response(hsm,vldinput_set,Ks) #predicted response for validation set

from visualization import *
corr = computeCorr(response, training_set)
vld_corr = computeCorr(pred_response,vld_set)


from funcs_for_graphs import *

report_txt="Training Set\nMean corr = %.4f, best neuron has corr = %.4f, median neuron=%.4f"%(corr.mean(), corr.max(), np.median(corr))
plot_act_of_max_min_corr(report_txt, response,training_set,corr, PLOT=True,ZOOM=True)

report_txt="Validation Set\nMean corr = %.4f, best neuron has corr = %.4f, median neuron=%.4f"%(vld_corr.mean(), vld_corr.max(), np.median(vld_corr))
plot_act_of_max_min_corr(report_txt, pred_response,vld_set,vld_corr, PLOT=True,ZOOM=False)

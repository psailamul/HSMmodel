from HSM import HSM
import numpy
from scipy.optimize import minimize
import param

import numpy as np
from fitting import fitHSM
import time
from get_host_path import get_host_path
import os
import sys
from visualization import *
from funcs_for_graphs import *

#python HSM_theano_SciPy_minimize_multirun.py REGION=1 RESTART_TRIAL=0 SEED=13 | tee Seed_LOG/HSM_Theano_region1_seed13_trial0.txt

def main():
    #########################################################################
    # Simulation Config
    ########################################################################
    PLOT_CORR_STATS = False
    REGION =1; RESTART_TRIAL=4; SEED =13; ITERATIONS=1; LR = 1e-3; NUM_LGN=9; HLSR=0.2;
    MAXITER = 100000
    if len(sys.argv) > 1:
        for ii in range(1,len(sys.argv)):
            arg = sys.argv[ii]
            print(arg)
            exec(arg) 
    #tf.set_random_seed(SEED)
    print('SEED : %g'%(SEED))
    tt_run_time = time.time()

    download_time = time.time()
    Region_num=str(REGION)
    seed=SEED; lgn=NUM_LGN; hlsr=HLSR
    runID = RESTART_TRIAL
    curr = os.getcwd()


    training_inputs=np.load(os.path.join(curr,'Data/region'+Region_num+'/training_inputs.npy'))
    training_set=np.load(os.path.join(curr,'Data/region'+Region_num+'/training_set.npy'))
    print "Download complete: Time %s" %(time.time() - download_time)
    call_time = time.time()


    num_pres,num_neurons = numpy.shape(training_set)
    print "Creating HSM model"
    hsm = HSM(training_inputs,training_set) # Initialize model --> add input and output, construct parameters , build mobel, # create loss function
    print "Created HSM model"   
    hsm.num_lgn = lgn 
    hsm.hlsr = hlsr
        
    func = hsm.func() 

    Ks = hsm.create_random_parametrization(seed) # set initial random values of the model parameter vector

    MAXITER=100000
    Code='SciPytestSeed'
    HOST, PATH = get_host_path(HOST=True, PATH=True)
    #SUMMARY_DIR = 'TFtrainingSummary/SciPy_maxiter_grad/'
    SUMMARY_DIR = 'TFtrainingSummary/SciPy_SEEDnumpy/'
    #c(Ks,success,c)=fmin_tnc(func ,Ks,fprime=hsm.der(),bounds=hsm.bounsd,maxfun = 100000,messages=0)  # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_tnc.html
    out=minimize(func ,Ks,method='TNC',jac=hsm.der(),bounds=hsm.bounds,options={'maxiter':MAXITER,'disp':True})

    Ks=out.x
    np.save("%sHSMout_theano_%s_Rg%s_MaxIter%g_seed%g-%g.npy"%(SUMMARY_DIR,Code,Region_num, MAXITER,seed,runID ), out, hsm) #NOTE: use savez in Final  code
    print "Saved"
    #import ipdb; ipdb.set_trace()
    print 'Final training error: ', func(numpy.array(Ks))/num_neurons/len(training_set)
    runtime = time.time() - call_time


    print "Finish training %s" %(runtime)
    if PLOT_CORR_STATS:
        raw_vld_set=np.load(os.path.join(curr,'Data/region'+Region_num+'/raw_validation_set.npy'))
        vldinput_set=np.load(os.path.join(curr,'Data/region'+Region_num+'/validation_inputs.npy'))
        vld_set=np.load(os.path.join(curr,'Data/region'+Region_num+'/validation_set.npy'))

        response = HSM.response(hsm,training_inputs,Ks) # predicted response after train

        #Validation set
        pred_response = HSM.response(hsm,vldinput_set,Ks) #predicted response for validation set

        corr = computeCorr(response, training_set)
        vld_corr = computeCorr(pred_response,vld_set)

        report_txt="Region #%s: Training Set\nMean corr = %.4f, best neuron has corr = %.4f, median neuron=%.4f"%(Region_num, corr.mean(), corr.max(), np.median(corr))
        plot_act_of_max_min_corr(report_txt, response,training_set,corr, PLOT=True,ZOOM=True)

        report_txt="Region #%s: Validation Set\nMean corr = %.4f, best neuron has corr = %.4f, median neuron=%.4f"%(Region_num, vld_corr.mean(), vld_corr.max(), np.median(vld_corr))
        plot_act_of_max_min_corr(report_txt, pred_response,vld_set,vld_corr, PLOT=True,ZOOM=False)

if __name__ == "__main__":
    main()
from HSM import HSM
import numpy
from scipy.optimize import minimize
import param
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fitting import fitHSM
import time
from get_host_path import get_host_path
from datetime import datetime
import os
import sys
import re
from visualization import *
from funcs_for_graphs import *

#python HSM_theano_SciPy_minimize_multirun.py REGION=1 RESTART_TRIAL=0 SEED=13 | tee Seed_LOG/HSM_Theano_region1_seed13_trial0.txt
def saveplot_act_of_max_min_corr(yhat,train_set,corr,save_dir,fname, setname='TR',report_txt=''):
    imax = np.argmax(corr) # note : actually have to combine neurons in all regions
    plt.figure()
    plt.plot(train_set[:,imax],'-ok')
    plt.plot(yhat[:,imax],'--or')
    plt.title('Cell#%d has max corr of %f\n%s'%(imax+1,np.max(corr),report_txt))
    plt.savefig('%s/STATmaxcorr_%s_%s.png'%(save_dir,setname, fname))

    imin = np.argmin(corr) # note : actually have to combine neurons in all regions
    plt.figure()
    plt.plot(train_set[:,imin],'-ok')
    plt.plot(yhat[:,imin],'--or')
    plt.title('Cell#%d has min corr of %f\n%s'%(imin+1,np.min(corr),report_txt))
    plt.savefig('%s/STATmincorr_%s_%s.png'%(save_dir,setname, fname))
    
def main():
    #########################################################################
    # Simulation Config
    ########################################################################
    dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    PLOT_CORR_STATS = True
    REGION =1; RESTART_TRIAL=0; SEED =13; ITERATIONS=1; LR = 1e-3; NUM_LGN=9; HLSR=0.2;
    MAXITER = 100000
    if len(sys.argv) > 1:
        for ii in range(1,len(sys.argv)):
            arg = sys.argv[ii]
            print(arg)
            exec(arg) 
    #tf.set_random_seed(SEED)
    print('SEED : %g'%(SEED))
    tt_run_time = time.time()
    Code='SciPytestSeed'
    HOST = 'ccv'
    
    download_time = time.time()
    Region_num=str(REGION)
    seed=SEED; lgn=NUM_LGN; hlsr=HLSR
    runID = RESTART_TRIAL

    DATA_LOC = "Data/" #"/users/psailamu/data/psailamu/AntolikData/"
    OUTPUT_LOC = "/users/psailamu/data/psailamu/HSMmodel/TFtrainingSummary/SciPy_SEEDnumpy/"
    code_for_this_run = "HSMout_theano_%s_Rg%s_MaxIter%g_seed%g-%g_%s"%(Code,
                                Region_num, 
                                MAXITER,
                                seed,
                                runID,
                                dt_stamp)
    
    figure_output_dir = OUTPUT_LOC+code_for_this_run   
    
    try: 
        os.makedirs(figure_output_dir)
    except OSError:
        if not os.path.isdir(figure_output_dir):
            raise
            
    
    training_inputs=np.load(os.path.join(DATA_LOC,'region'+Region_num+'/training_inputs.npy'))
    training_set=np.load(os.path.join(DATA_LOC,'region'+Region_num+'/training_set.npy'))
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

    #c(Ks,success,c)=fmin_tnc(func ,Ks,fprime=hsm.der(),bounds=hsm.bounsd,maxfun = 100000,messages=0)  # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_tnc.html
    out=minimize(func ,Ks,method='TNC',jac=hsm.der(),bounds=hsm.bounds,options={'maxiter':MAXITER,'disp':True})

    Ks=out.x
    np.save("%sHSMout_theano_%s_Rg%s_MaxIter%g_seed%g-%g.npy"%(OUTPUT_LOC,Code,Region_num, MAXITER,seed,runID ), out, hsm) #NOTE: use savez in Final  code
    print "Saved"
    #import ipdb; ipdb.set_trace()
    print 'Final training error: ', func(numpy.array(Ks))/num_neurons/len(training_set)
    runtime = time.time() - call_time


    print "Finish training %s" %(runtime)
    if PLOT_CORR_STATS:
        fname = "%s_Rg%s_MaxIter%g_seed%g-%g"%(Code,Region_num, MAXITER,seed,runID)
        raw_vld_set=np.load('Data/region'+Region_num+'/raw_validation_set.npy')
        vldinput_set=np.load('Data/region'+Region_num+'/validation_inputs.npy')
        vld_set=np.load('Data/region'+Region_num+'/validation_set.npy')

        response = HSM.response(hsm,training_inputs,Ks) # predicted response after train

        #Validation set
        pred_response = HSM.response(hsm,vldinput_set,Ks) #predicted response for validation set

        corr = computeCorr(response, training_set)
        vld_corr = computeCorr(pred_response,vld_set)

        report_txt="Region #%s: Training Set\nMean corr = %.4f, best neuron has corr = %.4f, median neuron=%.4f"%(Region_num, corr.mean(), corr.max(), np.median(corr))
        #plot_act_of_max_min_corr(report_txt, response,training_set,corr, PLOT=True,ZOOM=True)
        saveplot_act_of_max_min_corr(response,training_set,corr,figure_output_dir,fname, setname='TR',report_txt=report_txt)
        
        report_txt="Region #%s: Validation Set\nMean corr = %.4f, best neuron has corr = %.4f, median neuron=%.4f"%(Region_num, vld_corr.mean(), vld_corr.max(), np.median(vld_corr))
        #plot_act_of_max_min_corr(report_txt, pred_response,vld_set,vld_corr, PLOT=True,ZOOM=False)
        saveplot_act_of_max_min_corr(pred_response,vld_set,vld_corr,figure_output_dir,fname, setname='VLD',report_txt=report_txt)

if __name__ == "__main__":
    main()
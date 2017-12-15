import funcs_for_graphs as my_g

TNTF_TR_fig_max=my_g.compare_corr_all_regions(Theano_TR_pred_response,TF_TR_pred_response, TNTF_corr, stats_param='median', titletxt='Theano&TensorFlow :: Training Set', RETURN=True)
TNTF_VLD_fig_max=my_g.compare_corr_all_regions(Theano_VLD_pred_response,TF_VLD_pred_response, TNTF_vld_corr, stats_param='median', titletxt='Theano&TensorFlow :: Validation Set', RETURN=True)
vld_set
Theano_VLD_pred_response
TF_VLD_pred_response


stats_param='median'
titletxt='Validation Set'

corr_set=TNTF_vld_corr
stats_param='median'
corr1=corr_set['1']; corr2=corr_set['2']; corr3=corr_set['3']; 


    if stats_param.lower() == 'max' :
        idx1=np.argmax(corr1); idx2=np.argmax(corr2); idx3=np.argmax(corr3)
        stat1 = np.max(corr1); stat2 = np.max(corr2); stat3 = np.max(corr3)
    elif stats_param.lower() == 'min' :
        idx1=np.argmin(corr1); idx2=np.argmin(corr2); idx3=np.argmin(corr3)
        stat1 = np.min(corr1); stat2 = np.min(corr2); stat3 = np.min(corr3); 
    elif stats_param.lower() == 'median' :
        N1=len(corr1); N2=len(corr2); N3=len(corr3);
        idx1 = N1/2-1 if N1%2==0 else (N1-1)/2
        idx2 = N2/2-1 if N2%2==0 else (N2-1)/2
        idx3 = N3/2-1 if N3%2==0 else (N3-1)/2
        
        stat1=np.sort(corr1)[idx1]
        stat2=np.sort(corr2)[idx2]
        stat3=np.sort(corr3)[idx3]
    else:
        print "Parameter not Found "
    combine_corr = np.concatenate((corr1, corr2,corr3),axis=0)

    TN_R1=np.sort(TN_vld_corr['1'])[idx1]
    TN_R2=np.sort(TN_vld_corr['2'])[idx2]
    TN_R3=np.sort(TN_vld_corr['3'])[idx3]
    
    TF_R1=np.sort(TF_vld_corr['1'])[idx1]
    TF_R2=np.sort(TF_vld_corr['2'])[idx2]
    TF_R3=np.sort(TF_vld_corr['3'])[idx3]
    
    
    leg_loc =1
    fig, ax = plt.subplots()
    plt.subplot(311)
    plt.plot(vld_set['1'][:,idx1],'-ok',label='Measured Response')
    plt.plot(Theano_VLD_pred_response['1'][:, idx1],'--or',label="Antolik''s model, R =%.2f"%(TN_R1))
    plt.plot(TF_VLD_pred_response['1'][:, idx1],'--ob',label="My implementation, R =%.2f"%(TF_R1))
    plt.title('Cell#%g is the %s  neuron in region 1, R = %.5f, mean neuron has R = %.5f'%(idx1+1 ,stats_param,stat1, np.mean(corr1)))
    plt.legend(loc=leg_loc)
    
    plt.subplot(312)
    plt.plot(vld_set['2'][:,idx2],'-ok',label='Measured Response')
    plt.plot(Theano_VLD_pred_response['2'][:, idx2],'--or',label="Antolik''s model, R =%.2f"%(TN_R2))
    plt.plot(TF_VLD_pred_response['2'][:, idx2],'--ob',label="My implementation, R =%.2f"%(TF_R2))
    plt.title('Cell#%g is the %s neuron in region 2, R = %.5f, mean neuron has R =%.5f'%(idx2+1 ,stats_param,stat2, np.mean(corr2)))
    plt.legend(loc=leg_loc)
    
    plt.subplot(313)
    plt.plot(vld_set['3'][:,idx3],'-ok',label='Measured Response')
    plt.plot(Theano_VLD_pred_response['3'][:, idx3],'--or',label="Antolik''s model, R =%.2f"%(TN_R3))
    plt.plot(TF_VLD_pred_response['3'][:, idx3],'--ob',label="My implementation, R =%.2f"%(TF_R3))
    plt.title("Cell#%g is the %s neuron in region 3, R = %.5f, mean neuron has R = %.5f"%(idx3+1 ,stats_param,stat3, np.mean(corr3)))
    plt.legend(loc=leg_loc)
    
    report_txt="Overall mean corr = %.4f, best neuron has corr = %.4f, median neuron=%.4f"%(combine_corr.mean(), 
        combine_corr.max(), np.median(combine_corr))
            
    if titletxt=='':
        plt.suptitle(report_txt)
    else:
        plt.suptitle("%s\n%s"%(titletxt,report_txt))
    plt.show()
    
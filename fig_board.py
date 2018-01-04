


plot_fig4_response_scatter(model_activity=TN_VLD_pred_response_all, 
                cell_true_activity=vld_set_all, 
                corr_set=TN_vld_all_corr, 
                stats_param='max',
                filelog=figures_dir,
                filename ='TN_VLD_best_fig4a',
                runcodestr="Original HSM (mean R =%f )"%(np.mean(TN_vld_all_corr)),
                SAVE=True)
plot_fig4_response_scatter(model_activity=TN_VLD_pred_response_all, 
                cell_true_activity=vld_set_all, 
                corr_set=TN_vld_all_corr, 
                stats_param='median',
                filelog=figures_dir,
                filename ='TN_VLD_median_fig4a',
                runcodestr="Original HSM (mean R =%f )"%(np.mean(TN_vld_all_corr)),
                SAVE=True)
                
plot_fig4_response_scatter(model_activity=TF_VLD_pred_response_all, 
                cell_true_activity=vld_set_all, 
                corr_set=TF_vld_all_corr, 
                stats_param='max',
                filelog=figures_dir,
                filename ='TF_VLD_best_fig4a',
                runcodestr="Reimplemented HSM (mean R =%f )"%(np.mean(TF_vld_all_corr)),
                SAVE=True)
                
plot_fig4_response_scatter(model_activity=TF_VLD_pred_response_all, 
                cell_true_activity=vld_set_all, 
                corr_set=TF_vld_all_corr, 
                stats_param='median',
                filelog=figures_dir,
                filename ='TF_VLD_median_fig4a',
                runcodestr="Reimplemented HSM (mean R =%f )"%(np.mean(TF_vld_all_corr)),
                SAVE=True)

fileloc=figures_dir
filename ='TN_VLD_CDF'
fig_TNVLD,Xs,Fs = cdf_allregions( TN_vld_corr, NUM_REGIONS=3, DType="Original HSM (mean R =%f )\n"%(np.mean(TN_vld_all_corr)),fileloc=fileloc,filename=filename, C_CODE=True, SHOW=True, RETURN=True, SAVE=True)
filename ='TF_VLD_CDF'
fig_TFVLD,Xs,Fs = cdf_allregions( TF_vld_corr, NUM_REGIONS=3, DType="Reimplemented HSM (mean R =%f )\n"%(np.mean(TF_vld_all_corr)),fileloc=fileloc,filename=filename,  C_CODE=True, SHOW=True, RETURN=True, SAVE=True)


#TN & TF corr 
TNTF_corr={}; TNTF_vld_corr={}
for i in range(NUM_REGIONS):
    id=str(i+1)
    TNTF_corr[id] = computeCorr(Theano_TR_pred_response[id],TF_TR_pred_response[id])
    TNTF_vld_corr[id]=computeCorr(Theano_VLD_pred_response[id],TF_VLD_pred_response[id])

#Histogram of TN & TF correlation
combine_TNTF_corr = np.concatenate((TNTF_corr['1'], TNTF_corr['2'],TNTF_corr['3']),axis=0)
combine_TNTF_vld_corr = np.concatenate((TNTF_vld_corr['1'], TNTF_vld_corr['2'],TNTF_vld_corr['3']),axis=0)

filename ='TNTF_corr_hist'
plt.figure(figsize=[14,5])
weights = 100.0*np.ones_like(combine_TNTF_corr)/float(len(combine_TNTF_corr))
plt.subplot(121); plt.hist(combine_TNTF_corr,weights=weights); 
plt.xlim([0,1]); plt.title('Training set');
plt.xlabel('Correlation Coefficient')
plt.ylabel('% of neuron')
weights = 100.0*np.ones_like(combine_TNTF_vld_corr)/float(len(combine_TNTF_vld_corr))
plt.subplot(122); plt.hist(combine_TNTF_vld_corr,weights=weights); 
plt.xlim([0,1]); plt.title('Testing set'); 
plt.xlabel('Correlation Coefficient'); plt.ylabel('% of neuron')
plt.suptitle('Distribution of Pearson correlation coefficient between the responses  predicted by the original HSM model and the reimplemented version')
plt.savefig(os.path.join(fileloc,filename+'.svg'))
plt.savefig(os.path.join(fileloc,filename+'.png'))
plt.show()


# TN TF True
#TN_VLD_pred_response_all
#TF_VLD_pred_response_all
#vld_set_all
TNTF_all_vld_corr =concat_flatten_regions(TNTF_vld_corr)
max_cell = np.argmax(TNTF_all_vld_corr)



################### del
#All cells


Ccode=('k','b','#F97306')
all_TN = TN_VLD_pred_response_all.flatten()
all_TF = TF_VLD_pred_response_all.flatten()
r_sqr = TN_TF_Rsquare(all_TN, all_TF)
line_len = np.ceil(np.max([np.max(all_TN),np.max(all_TF)]))
plt.scatter(all_TN,all_TF, facecolors='none', edgecolors=Ccode[id])
plt.plot(np.arange(line_len+1),np.arange(line_len+1),'k')

plt.text(line_len-1, line_len-1, 'y = x',
         rotation=45,
         horizontalalignment='center',
         verticalalignment='top',
         multialignment='center')
plt.title("Predicted Responses from training set when seed = %g\nR^2 = %f"%(SEED,r_sqr))
plt.xlabel("Antolik's implementation (Theano)")
plt.ylabel("Re-implementation with Tensorflow")
plt.show()

R2_all={}
for s1 in seed_list:
  R2_all[s1] ={}
  for s2 in seed_list:
    setname ='validation'
    response1 =TN_VLD_pred_checkseed[s1]
    response2 =TF_VLD_pred_checkseed[s2]
    r_sqr = TN_TF_Rsquare(response1, response2)
    titletxt = "Predicted responses from %s set\nR^2 = %f"%(setname,r_sqr)
    xlbl="Theano Seed:%s"%s1
    ylbl="Tensorflow Seed:%s"%s2
    fhandle=plot_TN_TF_scatter_linear(response1, response2, titletxt=titletxt,xlbl=xlbl,ylbl=ylbl, RETURN=True)
    R2_all[s1][s2] = [TN_TF_Rsquare(this_TN, this_TF) for this_TN, this_TF in zip(response1.T,response2.T)]
    fname = "Check_seed_VLD_TN%s_TF%s"%(s1,s2)
    fhandle.savefig(Fig_fold+fname+'.png')
    plt.show()

corr_all={}
for s1 in seed_list:
  corr_all[s1] ={}
  for s2 in seed_list:
    setname ='validation'
    response1 =TN_VLD_pred_checkseed[s1]
    response2 =TF_VLD_pred_checkseed[s2]
    corr = computeCorr_flat(response1,response2)
    print "TN:%s TF:%s corr = %f"%(s1,s2,corr)
    corr_all[s1][s2] = corr


corr_TN={}
for s1 in seed_list:
  corr_TN[s1] ={}
  for s2 in seed_list:
    setname ='validation'
    response1 =TN_VLD_pred_checkseed[s1]
    response2 =TN_VLD_pred_checkseed[s2]
    corr = computeCorr_flat(response1,response2)
    print "TN:%s TN:%s corr = %f"%(s1,s2,corr)
    corr_TN[s1][s2] = corr
corr_TN={}
for s1 in seed_list:
  corr_TN[s1] ={}
  for s2 in seed_list:
    setname ='validation'
    response1 =TN_VLD_pred_checkseed[s1]
    response2 =TN_VLD_pred_checkseed[s2]
    corr = computeCorr(response1,response2)
    print "TN:%s TN:%s corr = %f"%(s1,s2,corr.mean())
    corr_TN[s1][s2] = corr.mean()

corr_TF={}
for s1 in seed_list:
  corr_TF[s1] ={}
  for s2 in seed_list:
    setname ='validation'
    response1 =TF_VLD_pred_checkseed[s1]
    response2 =TF_VLD_pred_checkseed[s2]
    corr = computeCorr_flat(response1,response2)
    print "TF:%s TF:%s corr = %f"%(s1,s2,corr)
    corr_TF[s1][s2] = corr
    

plot_fig4_response_scatter(model_activity=TF_VLD_pred_response_all, 
                cell_true_activity=TN_VLD_pred_response_all, 
                corr_set=TNTF_all_vld_corr, 
                stats_param='max',
                filelog=figures_dir,
                filename ='TNTF_VLD',
                runcodestr="TN-TF(mean R =%f )"%(np.mean(TNTF_all_vld_corr)),
                SAVE=False
                )
def plot_fig4_response_scatter( model_activity, 
                                cell_true_activity, 
                                corr_set, 
                                stats_param='max',
                                fileloc='',
                                filename='fig4',
                                runcodestr='',
                                datalabel1='Recorded neural response', 
                                datalabel2='Predicted response from model',
                                RETURN=False, 
                                SAVE=False
                                ):
    
    if stats_param.lower() == 'max' :
        idx1=np.argmax(corr_set);
        stat1 = np.max(corr_set);
        nr_text ='best'
    elif stats_param.lower() == 'min' :
        idx1=np.argmin(corr_set);
        stat1 = np.min(corr_set);
        nr_text ='worst'
    elif stats_param.lower() == 'median' or stats_param.lower() == 'med':
        N1=len(corr_set);
        idx1 = N1/2-1 if N1%2==0 else (N1-1)/2
        stat1=np.sort(corr_set)[idx1]
        nr_text ='median'
    else:
        print "Parameter not Found "
    
    #Compare response
    fig, ax = plt.subplots(figsize=[16,5])
    plt.subplot(1,2,1)
    plt.plot(cell_true_activity[:,idx],'-ok',label=datalabel1)
    plt.plot(model_activity[:,idx], '--ok', markerfacecolor='white',label=datalabel2)
    ylim_up = np.ceil(np.max([np.max(cell_true_activity[:,idx]),np.max(model_activity[:,idx])]))
    ylim_low = np.floor(np.min([np.min(cell_true_activity[:,idx]),np.min(model_activity[:,idx])]))
    plt.ylim([ylim_low,ylim_up])
    plt.xlabel('Image #')
    plt.ylabel('Response')
    plt.legend(loc=0)

    #scatter plot
    plt.subplot(1,2,2)
    plt.scatter(cell_true_activity[:,idx1], model_activity[:,idx1], facecolors='none', edgecolors='k')
    N=np.ceil(np.max([np.max(model_activity[:,idx1]),np.max(cell_true_activity[:,idx1])]))
    plt.plot(np.arange(N),np.arange(N),'--',color=(0.6,0.6,0.6),label='reference line(y = x)')
    plt.xlim([0,N]); plt.ylim([0,N])
    plt.ylabel(datalabel2)
    plt.xlabel(datalabel1)
    plt.legend(loc=0)
    plt.suptitle('%s\nResponse per image of the %s neuron, R=%f'%(runcodestr,nr_text,stat1))
    if SAVE:
        plt.savefig(os.path.join(fileloc,filename+'.svg'))
        plt.savefig(os.path.join(fileloc,filename+'.png'))
    plt.show()
    
    if RETURN :
        return fig
 
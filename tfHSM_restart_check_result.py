# load training of all restart then analyzed



def correlate_vectors(yhat_array, y_array):
  corrs = []      
  for yhat, y in zip(np.transpose(yhat_array), np.transpose(y_array)):
    tc = np.corrcoef(yhat, y)[1,0]
    if np.isnan(tc):
      tc=0.0
    #tc = (np.isnan(tc) == False).astype(float) * tc
    corrs += [tc]
  return corrs

def log_likelihood(predictions,targets,epsilon =0.0000000000000000001):
  return tf.reduce_sum(predictions) - tf.reduce_sum(tf.mul(targets,tf.log(predictions + epsilon)))

def hist_of_pred_and_record_response(pred_response, recorded_response, cell_id=0):
  plt.subplot(121); plt.hist(recorded_response[:,cell_id]); plt.title('Recorded Response');
  plt.subplot(122); plt.hist(pred_response[:,cell_id]); plt.title('Predicted Response');
  plt.suptitle("Distribution of cell #%g's response"%cell_id)
  plt.show()

def plot_act_of_max_min_corr(yhat,train_set,corr):
    imax = np.argmax(corr) # note : actually have to combine neurons in all regions

    plt.plot(train_set[:,imax],'-ok')
    plt.plot(yhat[:,imax],'--or')
    plt.title('Cell#%d has max corr of %f'%(imax+1,np.max(corr)))
    plt.show()

    imin = np.argmin(corr) # note : actually have to combine neurons in all regions

    plt.plot(train_set[:,imin],'-ok')
    plt.plot(yhat[:,imin],'--or')
    plt.title('Cell#%d has min corr of %f'%(imin+1,np.min(corr)))
    plt.show()

    
    
    #save
    if(SAVEdat):
        np.savez('%s/TRdat_%s.npz'%(summary_dir,summary_fname), 
         TR_loss=loss_list, 
         TR_mean_LGNact=activation_summary_lgn, 
         TR_mean_L1act=activation_summary_l1, 
         TR_std_pred_response =yhat_std, 
         TR_MSE=MSE_list, 
         TR_corr=corr_list,
         TR_last_pred_response=yhat,
         TR_last_l1_response=l1_response,
         TR_last_lgn_response=lgn_response,
         CONFIG=CONFIG)
    
    
    # check the training
    if(VISUALIZE):
        itr_idx = range(iterations)
        plt.subplot(2, 3, 1)
        plt.plot(itr_idx, loss_list, 'k-')
        plt.title('Loss')
        plt.xlabel('iterations')

        plt.subplot(2, 3, 2)
        plt.plot(itr_idx, MSE_list, 'b-')
        plt.title('MSE')
        plt.xlabel('iterations')

        plt.subplot(2, 3, 3)
        plt.plot(itr_idx, corr_list, 'r-')
        plt.title('Mean Correlation')
        plt.xlabel('iterations')

        plt.subplot(2, 3, 4)
        plt.plot(itr_idx, activation_summary_lgn, 'r-')
        plt.title('Mean LGN activation')
        plt.xlabel('iterations')

        plt.subplot(2, 3, 5)
        plt.plot(itr_idx, activation_summary_l1, 'r-')
        plt.title('Mean L1 activation')
        plt.xlabel('iterations')

        plt.subplot(2,3, 6)
        plt.plot(itr_idx, yhat_std, 'r-')
        plt.title('std of predicted response')
        plt.xlabel('iterations')

        plt.suptitle("Code: %s lr=%.5f , itr = %g\n[%s]"%(runcodestr,lr,iterations,str(datetime.now())))
        plt.show()

    if (PLOT_CORR_STATS):
        pred_act = yhat; responses = train_set
        #hist_of_pred_and_record_response(pred_act,responses)

        corr = computeCorr(yhat, train_set)
        corr[np.isnan(corr)]=0.0
        plot_act_of_max_min_corr(pred_act,responses,corr)
        hist_of_pred_and_record_response(pred_act,responses,cell_id=np.argmax(corr))
    print('Finished everything :: %s \n Time =%s '%(runcodestr,time.time() - tt_run_time))
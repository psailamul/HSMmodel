import numpy as np
import os

"""
[u'/home/pachaya/HSMmodel/Data/region1/raw_validation_set.npy',
 u'/home/pachaya/HSMmodel/Data/region1/training_inputs.npy',
 u'/home/pachaya/HSMmodel/Data/region1/validation_inputs.npy',
 u'/home/pachaya/HSMmodel/Data/region1/validation_set.npy',
 u'/home/pachaya/HSMmodel/Data/region1/training_set.npy']
"""

data_dir = u'/home/pachaya/HSMmodel/Data/region1'
main_data_fold = u'/home/pachaya/HSMmodel/Data/'
train_input = np.load(os.path.join(data_dir,"training_inputs.npy"))
train_output = np.load(os.path.join(data_dir,"training_set.npy"))

num_im = train_input.shape[0]
num_im_list = [int(x*num_im) for x in [0.05, 0.1, 0.25, 0.50, 0.75, 1.0]] # [90, 180, 450, 900, 1350, 1800] 


new_fold_list = ["region%g"%x for x in num_im_list]
#create new folder
for fold in new_fold_list:
	os.system("mkdir %s%s"%(main_data_fold,fold))
#save training set
for num_x in num_im_list:
	fold = "region%g"%num_x
	tmp_train_input = train_input[:int(num_x),:]
	tmp_train_out = train_output[:int(num_x),:]
	np.save(os.path.join(main_data_fold,fold,"training_inputs.npy"),tmp_train_input)
	np.save(os.path.join(main_data_fold,fold,"training_set.npy"),tmp_train_out)
#copy validation set 
vldin = '/home/pachaya/HSMmodel/Data/region1/validation_inputs.npy'
vldout =  '/home/pachaya/HSMmodel/Data/region1/validation_set.npy'
rawvld = '/home/pachaya/HSMmodel/Data/region1/raw_validation_set.npy'
for fold in new_fold_list:
	os.system("cp %s %s%s/"%(vldin, main_data_fold,fold))
	os.system("cp %s %s%s/"%(vldout, main_data_fold,fold))
	os.system("cp %s %s%s/"%(rawvld, main_data_fold,fold))
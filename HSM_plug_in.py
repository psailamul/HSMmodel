# Rebuild_filter



import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import tensorflow as tf 
import numpy as np

#Load learned model 
reg1 = np.load('out_region1.npy')

img_size =31
img_vec_size = img_size**2


def DoG_weight(ks):
  # Passing the parameters for a LGN neuron
  x, y, sc, ss, rc, rs = ks
  grid_xx, grid_yy = np.meshgrid(np.arange(img_size),np.arange(img_size))
  grid_xx = np.reshape(grid_xx, [img_vec_size])
  grid_yy = np.reshape(grid_yy, [img_vec_size])
  pi = np.pi
  pos = ((grid_xx - x)**2 + (grid_yy - y)**2)
  center = np.exp(-pos/2/sc) / (2*(sc)*pi)
  surround = np.exp(-pos/2/(sc + ss)) / (2*(sc + ss)*pi)
  DoGweight = (rc*(center)) - (rs*(surround))
  DoGweight = DoGweight.reshape([img_size,img_size])
  return DoGweight

k = reg1[0]

for ii in range(9):
    lgn_param = k[6*ii:6*ii+6]
    x, y, sc, ss, rc, rs = lgn_param
    print "%g:\n\tx,y = (%.2f,%.2f)\n\tsc = %.2f , ss = %.2f\n\trc = %.2f, rs = %.2f"%(ii,x,y,sc,ss,rc,rs)
    dog_w = DoG_weight(lgn_param)
    imgplot = plt.imshow(dog_w)
    plt.colorbar()
    plt.title("%g"%ii)
    plt.show()



# For TensorFlow --- TF summary
from tf_HSM_upgraded_SciPy_Numpy import tf_HSM

meta = '/home/pachaya/HSMmodel/TFtrainingSummary/SciPy_SEEDnumpy/AntolikRegion3_SciPy_jac_npSeed_MaxIter100000_itr1_SEED13_2017_12_13_15_59_54/trainedHSM_region3_trial13-0.meta'
saver = tf.train.import_meta_graph(meta) #Load meta graph

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
  sess.run(fetch, 'w1:0')
LGN_params



#

from get_host_path import get_host_path
PATH = get_host_path(HOST=False,PATH=True)
training_inputs=np.load(PATH+'Data/region'+Region_num+'/training_inputs.npy')
training_set=np.load(PATH+'Data/region'+Region_num+'/training_set.npy')

fig, ax = plt.subplots()
num_img = training_inputs.shape[0]
for i in range(num_img):
  ax.cla()
  this_img = training_inputs[i,:]
  this_img = this_img.reshape([31,31])
  ax.imshow(this_img,cmap='gray')
  ax.set_title(i)
  plt.pause(0.1)



fig, ax = plt.subplots()

for i in range(len(data)):
    ax.cla()
    ax.imshow(data[i])
    ax.set_title("frame {}".format(i))
    # Note that using time.sleep does *not* work here!
    plt.pause(0.1)
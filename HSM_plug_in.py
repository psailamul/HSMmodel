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


for ii in range(9):
    lgn_param = k[6*ii:6*ii+6]
    dog_w = DoG_weight(lgn_param)
    imgplot = plt.imshow(dog_w)
    plt.title("%g"%ii)
    plt.show()

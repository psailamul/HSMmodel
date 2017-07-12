#TensorFlow
print "==TensorFlow=="
import tensorflow as tf
tf.set_random_seed(13)
c = tf.random_uniform([], -10, 10)
d = tf.random_uniform([], -10, 10)
with tf.Session() as sess:
  print sess.run(c)
  print sess.run(d)

#numpy 
print "==Numpy=="
import numpy as np
from numpy.random import seed
seed(13)
c=np.random.uniform(low=-10,high=10)
d=np.random.uniform(low=-10,high=10)
print c
print d
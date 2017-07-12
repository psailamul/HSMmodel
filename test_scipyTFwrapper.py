
import tensorflow as tf
ScipyOptimizerInterface = tf.contrib.opt.ScipyOptimizerInterface

# Scipy and TF wrapper tutorial
vector = tf.Variable([7., 7.], 'vector')

# Make vector norm as small as possible.
loss = tf.reduce_sum(tf.square(vector))

optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Need to initialize both of these if supplying num_epochs to inputs
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

with tf.Session() as session:
  optimizer.minimize(session)

import ipdb; ipdb.set_trace()
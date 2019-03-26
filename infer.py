import tensorflow as tf 
import skimage.io as io
import numpy as np 
import sys

import net

#Input placeholders
x = tf.placeholder(tf.float32, [None, 32, 32, 3])

#Predictor
pred = net.pred(x)

#Image path
path = str(sys.argv[-1])

#Image load
img = io.imread(path)
img_r = np.reshape(img, [1, img.shape[0], img.shape[1], 3])

#Saver 
saver = tf.train.Saver()

#Session
sess = tf.InteractiveSession()

#Load model
mdl_path = './models/weightV1.ckpt'
saver.restore(sess, mdl_path)

out = sess.run(pred, feed_dict = {x : img_r, net.tf_prob : 1.0})

classes = out[1]

io.imshow(classes[0])
io.show()
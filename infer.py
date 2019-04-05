import tensorflow as tf 
import skimage.io as io
import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib import patches 
import sys

import netv2 as net

#Input placeholders
x = tf.placeholder(tf.float32, [None, None, None, 3])

#Predictor
pred = net.pred(x)

#Image path
path = str(sys.argv[-1])

#Image load
img = io.imread(path)
pad = 0 #cSize // 2 + 5
img = np.pad(img, pad, mode = 'edge')[:, :, pad : pad + 3]
img_r = np.reshape(img, [1, img.shape[0], img.shape[1], 3])

#Saver 
saver = tf.train.Saver()

#Session
sess = tf.InteractiveSession()

#Load model
mdl_path = './models/weightV1.ckpt'
saver.restore(sess, mdl_path)

out = sess.run(pred, feed_dict = {x : img_r, net.tf_prob : 1.0})

detection = out[2] 
box = out[4][0, :, :, :]

#window size
window = 32 #pixels 
strides = 2 #strides 

#positional matrix 
shape = detection[0, :, :, 0].shape
p_arr = np.ones([shape[0], shape[1], 5])

for i, r in enumerate(detection[0, :, :, 0]):
        p_v = (i - 1) * strides + window // 2
        for j, c in enumerate(r):
                p_h = (j - 1) * strides + window // 2
                p_arr[i, j, 0] = p_v + box[i, j, 0]
                p_arr[i, j, 1] = p_h + box[i, j, 1]
                p_arr[i, j, 2] = p_v + box[i, j, 2]
                p_arr[i, j, 3] = p_h + box[i, j, 3]
                p_arr[i, j, 4] = c

fig, ax = plt.subplots(1)

ax.imshow(img)
x = 5
for r in p_arr[10 + x : 15 + x]:
        for c in r[10:15]:
                if c[-1] > 0.999:
                        rec = patches.Rectangle(c[:2], c[2] - c[0], c[3] - c[1], linewidth = 1, edgecolor='b', facecolor='none')
                        ax.add_patch(rec)

plt.show()

import tensorflow as tf
from skimage import transform as tr 
from skimage import io 
import pandas as pd 
import numpy as np 

import net

#fetch data 
def datFetch(dat, path, batch_size, iteration = 0, offset = 0, aug_chance = 0.3):
    imgs = []
    bboxes = []
    categories = []

    indexes = np.array(range(batch_size)) + iteration * batch_size + offset
    for index in indexes:
        
        #Image
        img = io.imread('./%s/%s' % (path, dat.loc[index][1])) / 255
        r = np.random.uniform() #Random float between 0-1
        if r <= aug_chance:
            img = augment(img)
        imgs.append(img.flatten())

        #Bounding box
        bbox = np.array(dat.loc[index][2][1 : -1])
        bboxes.append(bbox)

        #Category 
        onehot = np.identity(3)
        categories.append(onehot[dat.loc[index][3]])
    
    return np.array(imgs), np.array(bboxes), np.array(categories)
    
#augment data
def augment(x):
    angle = np.random.uniform() * 30 # max rotation angle
    noise = np.random.uniform(size = x.shape) * np.random.uniform() * 0.5
    return tr.rotate(x, angle, mode = 'edge') + noise

#placeholders 
#input
x = tf.placeholder(tf.float32, shape = [None, 32 * 32 * 3])
x_reshaped = tf.reshape(x, shape = [-1, 32, 32, 3])

#category
s = tf.placeholder(tf.float32, shape = [None, 3])
s_reshaped = tf.reshape(s, shape = [-1, 1, 1, 3])

#bounding box
b = tf.placeholder(tf.float32, shape = [None, 4])
b_reshped = tf.reshape(b, shape = [-1, 1, 1, 4])

logits_s, out_s, out_b = net.pred(x_reshaped) # outputs

#Losses 
#category 
s_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = s_reshaped, logits = logits_s)

#bounding box
c_loss = tf.losses.mean_squared_error(labels = b_reshped, predictions = out_b)

#Learning rate 
lr = 0.0015
#Optimizer 
opt = tf.train.AdamOptimizer(learning_rate = lr)

#Optimize ops
s_opt = opt.minimize(s_loss)
c_opt = opt.minimize(c_loss)
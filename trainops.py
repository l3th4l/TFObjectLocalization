import tensorflow as tf
from skimage import transform as tr 
from skimage import io 
import pandas as pd 
import numpy as np 
import re

import net

#fetch data 
def datFetch(dat, path, batch_size, iteration = 0, offset = 0, aug_chance = 0.3, load_box = True):
    imgs = []
    bboxes = []
    categories = []

    indexes = np.array(range(batch_size)) + iteration * batch_size + offset
    for index in indexes:
        
        #Image
        img = io.imread('./%s/%s' % (path, dat.loc[index]['filename'])) / 255
        r = np.random.uniform() #Random float between 0-1
        if r <= aug_chance:
            img = augment(img)
        imgs.append(img.flatten())

        #Category 
        onehot = np.identity(3)
        categories.append(onehot[int(dat.loc[index]['category'])])
        
        #Bounding box
        if load_box:
            bbox = np.array(re.sub(' +', ' ', dat.loc[index]['bounding_box'][1 : -2]).split(' ')[:4], np.float)
            bboxes.append(bbox)

        '''
        try:            
            bbox = np.array(re.sub(' +', ' ', dat.loc[index]['bounding_box'][1 : -2]).split(' '), np.float)
            bboxes.append(bbox)
        except:
            pass'''

    
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

#'1' tensors
s_ones = tf.ones_like(logits_s)
b_ones = tf.ones_like(out_b)

#Modified labels
s_mod = tf.multiply(s_reshaped, s_ones)
b_mod = tf.multiply(b_reshped, b_ones)

#Losses 
#category 
s_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = s_mod, logits = logits_s)

#bounding box
b_loss = tf.losses.mean_squared_error(labels = b_mod, predictions = out_b)

#Learning rate 
lr = 0.00001

#Optimizer 
opt = tf.train.AdamOptimizer(learning_rate = lr)
#opt = tf.train.GradientDescentOptimizer(learning_rate = lr)

#Optimize ops
s_opt = opt.minimize(s_loss)
b_opt = opt.minimize(b_loss)
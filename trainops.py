import tensorflow as tf
from skimage import transform as tr 
from skimage import io 
import pandas as pd 
import numpy as np 
import re

import netv2 as net

#fetch data 
def datFetch(dat, path, batch_size, iteration = 0, offset = 0, aug_chance = 0.3, load_box = True):
    imgs = []
    bboxes = []
    categories = []
    detection = []

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
        onehot[0, 0] = 0
        onehot = onehot[1 : , 1 : ]
        try:
            categories.append(onehot[int(dat.loc[index]['category']) - 1])
            detection.append(1)
        except:
            detection.append(0)
        
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

    
    return np.array(detection),  np.array(imgs), np.array(bboxes), np.array(categories)
    
#augment data
def augment(x):
    angle = np.random.uniform() * 30 # max rotation angle
    noise = np.random.uniform(size = x.shape) * np.random.uniform() * 0.5
    return tr.rotate(x, angle, mode = 'edge') + noise

#placeholders 
#input
x = tf.placeholder(tf.float32, shape = [None, 64 * 64 * 3])
x_reshaped = tf.reshape(x, shape = [-1, 64, 64, 3])

#obj detection
d = tf.placeholder(tf.float32, shape = [None])
d_reshaped = tf.reshape(d, shape = [-1, 1, 1, 1])

#category
c = tf.placeholder(tf.float32, shape = [None, 2])
c_reshaped = tf.reshape(c, shape = [-1, 1, 1, 2])

#bounding box
b = tf.placeholder(tf.float32, shape = [None, 4])
b_reshaped = tf.reshape(b, shape = [-1, 1, 1, 4])

logit_d, logit_c, out_d, out_c, out_b = net.pred(x_reshaped) # outputs


#Losses 

#obj detection
d_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = d_reshaped, logits = logit_d)
#category 
c_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = c_reshaped, logits = logit_c)
#bounding box
b_loss = tf.losses.mean_squared_error(labels = b_reshaped, predictions = out_b)

#Learning rate 
lr = 0.0001

#Optimizer 
opt = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5)
#opt = tf.train.GradientDescentOptimizer(learning_rate = lr)

#Optimize ops
d_opt = opt.minimize(d_loss)
c_opt = opt.minimize(c_loss)
b_opt = opt.minimize(b_loss)
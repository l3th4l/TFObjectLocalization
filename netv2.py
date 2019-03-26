import tensorflow as tf 
from tensorflow import keras as k

def cnv2d(x, kernel_size, channels, stride = [1, 1], padding = 'valid', name = 'c'):
    with tf.variable_scope(name):
        return tf.layers.conv2d(x, channels, kernel_size, strides = stride, padding = padding)

def pool(x, pool_size, stride = [2, 2], padding = 'valid', name = 'p'):
    with tf.variable_scope(name):
        return tf.layers.max_pooling2d(x, pool_size, strides = stride)

tf_prob = tf.placeholder_with_default(5.0, ())

def pred(x, scope = 'yolo'):

    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):

        cnv1 = cnv2d(x, [5, 5], 16, name = 'c1')                                                # --> 28 x 28 x 16
        pool1 = pool(cnv1, [2, 2], name = 'p1')                                                 # --> 14 x 14 x 16 

        cnv2 = cnv2d(pool1, [14, 14], 400, name = 'c2')                                         # --> 1 x 1 x 400
        fc1 = cnv2d(cnv2, [1, 1], 400, name = 'fc1')                                            # --> 1 x 1 x 400
        fc1 = k.layers.Dropout(tf_prob)(fc1)                                                    # dropout
        
        #---------------------------------------------------------------------------------------

        #Obj detection        

        d_fc1 = cnv2d(fc1, [1, 1], 32, name = 'dfc1')                                           # --> 1 x 1 x 32

        d_fc1 = tf.layers.batch_normalization(d_fc1)                                            # batch norm

        d_logit = cnv2d(d_fc1, [1, 1], 1, name = 'dl')                                          # --> 1 x 1 x 1

        d_out = tf.nn.sigmoid(d_logit, name = 'do')
        
        #Classification

        c_fc1 = cnv2d(fc1, [1, 1], 32, name = 'cfc1')                                           # --> 2 x 2 x 32

        c_fc1 = tf.layers.batch_normalization(c_fc1)                                            # batch norm

        c_logit = cnv2d(c_fc1, [1, 1], 2, name = 'cl')                                          # --> 2 x 2 x 2

        c_out = tf.nn.softmax(c_logit, name = 'co')

        #---------------------------------------------------------------------------------------

        #Bounding box        
        b_fc1 = cnv2d(fc1, [1, 1], 32, name = 'bfc1')                                           # --> 2 x 2 x 128
        
        b_fc1 = tf.layers.batch_normalization(b_fc1)                                            # batch norm

        b_out = cnv2d(b_fc1, [1, 1], 4, name = 'bc3')                                           # --> 2 x 2 x 32

    return d_logit, c_logit, d_out, c_out, b_out
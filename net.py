import tensorflow as tf 
from tensorflow import keras as k

def cnv2d(x, kernel_size, channels, stride = [1, 1], padding = 'same', name = 'c'):
    with tf.variable_scope(name):
        return tf.layers.conv2d(x, channels, kernel_size, strides = stride, padding = padding)

def pool(x, pool_size, stride = [2, 2], padding = 'same', name = 'p'):
    with tf.variable_scope(name):
        return tf.layers.max_pooling2d(x, pool_size, strides = stride)

tf_prob = tf.placeholder_with_default(5.0, ())

def pred(x, scope = 'yolo'):

    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):

        cnv1 = cnv2d(x, [4, 4], 16, name = 'c1')                                                # --> 32 x 32 x 16
        cnv2 = cnv2d(cnv1, [4, 4], 32, name = 'c2')                                             # --> 32 x 32 x 32

        cnv2 = k.layers.Dropout(tf_prob)(cnv2)                                                  # dropout

        cnv2 = tf.layers.batch_normalization(cnv2)                                              # batch norm

        pool1 = pool(cnv2, [4, 4], name = 'p1')                                                 # --> 16 x 16 x 32

        cnv3 = cnv2d(pool1 , [4, 4], 64, name = 'c3')                                           # --> 16 x 16 x 64
        cnv4 = cnv2d(cnv3, [4, 4], 128, name = 'c4')                                            # --> 16 x 16 x 128

        cnv4 = k.layers.Dropout(tf_prob)(cnv4)                                                  # dropout

        cnv4 = tf.layers.batch_normalization(cnv4)                                              # batch norm

        pool2 = pool(cnv4, [4, 4], name = 'p2')                                                 # --> 8 x 8 x 128

        cnv5 = cnv2d(pool2, [4, 4], 256, name = 'c5')                                           # --> 8 x 8 x 256
        cnv6 = cnv2d(cnv5, [4, 4], 512, name = 'c6')                                            # --> 8 x 8 x 512

        cnv6 = k.layers.Dropout(tf_prob)(cnv6)                                                  # dropout
        
        cnv6 = tf.layers.batch_normalization(cnv6)                                              # batch norm

        pool3 = pool(cnv6, [4, 4], name = 'p3')                                                 # --> 4 x 4 x 512

        #---------------------------------------------------------------------------------------

        #Segmentation 
        s_cnv1 = cnv2d(pool3, [4, 4], 128, [2, 2], name = 'sc1')                                # --> 2 x 2 x 128
        s_cnv2 = cnv2d(s_cnv1, [1, 1], 64, name = 'sc2')                                        # --> 2 x 2 x 64

        #Obj detection
        
        d_cnv2 = k.layers.Dropout(tf_prob)(s_cnv2)                                              # dropout

        d_cnv3 = cnv2d(d_cnv2, [1, 1], 32, name = 'sc3')                                        # --> 2 x 2 x 32

        d_cnv3 = tf.layers.batch_normalization(d_cnv3)                                          # batch norm

        d_logit = cnv2d(d_cnv3, [1, 1], 2, name = 'sl1')                                        # --> 2 x 2 x 3

        d_out = tf.nn.sigmoid(d_logit, name = 'so1')
        
        #Classification

        c_cnv2 = k.layers.Dropout(tf_prob)(s_cnv2)                                              # dropout

        c_cnv3 = cnv2d(c_cnv2, [1, 1], 32, name = 'sc3')                                        # --> 2 x 2 x 32

        c_cnv3 = tf.layers.batch_normalization(c_cnv3)                                          # batch norm

        c_logit = cnv2d(c_cnv3, [1, 1], 2, name = 'sl1')                                        # --> 2 x 2 x 3

        c_out = tf.nn.softmax(c_logit, name = 'so1')

        #---------------------------------------------------------------------------------------

        #Bounding box        
        b_cnv1 = cnv2d(pool3, [4, 4], 128, [2, 2], name = 'bc1')                                # --> 2 x 2 x 128
        b_cnv2 = cnv2d(b_cnv1, [1, 1], 64, name = 'bc2')                                        # --> 2 x 2 x 64

        b_cnv2 = k.layers.Dropout(tf_prob)(b_cnv2)                                              # dropout
        
        b_cnv2 = tf.layers.batch_normalization(b_cnv2)                                          # batch norm

        b_cnv3 = cnv2d(b_cnv2, [1, 1], 32, name = 'bc3')                                        # --> 2 x 2 x 32
        b_out = cnv2d(b_cnv3, [1, 1], 4, name = 'bo')                                           # --> 2 x 2 x 2

    return d_logit, c_logit, d_out, c_out, b_out
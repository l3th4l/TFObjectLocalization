import tensorflow as tf 

def cnv2d(x, kernel_size, channels, stride = [1, 1], padding = 'same', name = 'c'):
    with tf.variable_scope(name):
        return tf.layers.conv2d(x, channels, kernel_size, strides = stride, padding = padding)

def pool(x, pool_size, stride = [2, 2], padding = 'same', name = 'p'):
    with tf.variable_scope(name):
        return tf.layers.max_pooling2d(x, pool_size, strides = stride)

def pred(self, x, scope = 'yolo'):

    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):

        cnv1 = cnv2d(x, [4, 4], 16, name = 'c1') # --> 32 x 32 x 16
        cnv2 = cnv2d(cnv1, [4, 4], 32, name = 'c2') # --> 32 x 32 x 32

        pool1 = pool(cnv2, [4, 4], name = 'p1') # --> 16 x 16 x 32

        cnv3 = cnv2d(pool1 , [4, 4], 64, name = 'c3') # --> 16 x 16 x 64
        cnv4 = cnv2d(cnv3, [4, 4], 128, name = 'c4') # --> 16 x 16 x 128

        pool2 = pool(cnv4, [4, 4], name = 'p2') # --> 8 x 8 x 128

        cnv5 = cnv2d(pool2, [4, 4], 256, name = 'c5') # --> 8 x 8 x 256
        cnv6 = cnv2d(cnv5, [4, 4], 512, name = 'c6') # --> 8 x 8 x 512

        pool3 = pool(cnv6, [4, 4], name = 'p3') # --> 4 x 4 x 512

        #Segmentation 
        s_cnv1 = cnv2d(pool3, [4, 4], 128, [2, 2], 'valid', name = 'sc1') # --> 1 x 1 x 128
        s_cnv2 = cnv2d(s_cnv1, [1, 1], 64, name = 'sc2') # --> 1 x 1 x 64
        s_cnv3 = cnv2d(s_cnv2, [1, 1], 32, name = 'sc3') # --> 1 x 1 x 32
        s_logit = cnv2d(s_cnv3, [1, 1], 2, name = 'sl') # --> 1 x 1 x 2

        s_out = tf.nn.softmax(s_logit, name = 'so')

        #Bounding box        
        b_cnv1 = cnv2d(pool3, [4, 4], 128, [2, 2], 'valid', name = 'bc1') # --> 1 x 1 x 128
        b_cnv2 = cnv2d(b_cnv1, [1, 1], 64, name = 'bc2') # --> 1 x 1 x 64
        b_cnv3 = cnv2d(b_cnv2, [1, 1], 32, name = 'bc3') # --> 1 x 1 x 32
        b_out = cnv2d(b_cnv3, [1, 1], 4, name = 'bo') # --> 1 x 1 x 2

    return s_logit, s_out, b_out
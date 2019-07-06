import tensorflow as tf
import numpy as np

REGULARIZER_COF = 1e-8

def _fc_variable( weight_shape,name="fc"):
    with tf.variable_scope(name):
        # check weight_shape
        input_channels  = int(weight_shape[0])
        output_channels = int(weight_shape[1])
        weight_shape    = (input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)

        # define variables
        weight = tf.get_variable("_w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer =regularizer)
        bias   = tf.get_variable("_b", [weight_shape[1]],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv_variable( weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.constant_initializer(0.0),
                                regularizer=regularizer)

        bias   = tf.get_variable("b", [output_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv2d( x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def _flatten(x):
    n_b, n_h, n_w, n_f = [int(s) for s in x.get_shape()]
    h = tf.reshape(x,[n_b,n_h*n_w*n_f])
    return h

def _fc_layer(x, input_layer, output_layer, name="fc", isTraining=True):
    w, b = _fc_variable([input_layer,output_layer],name=name)
    h = tf.matmul(x, w) + b
    return h

def _conv_layer(x, input_layer, output_layer, stride, filter_size=3, name="conv", isTraining=True):
    conv_w, conv_b = _conv_variable([filter_size,filter_size,input_layer,output_layer],name=name)
    h = _conv2d(x,conv_w,stride=stride) + conv_b
    return h

def _max_pooling(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME"):
    return tf.nn.max_pool(x,ksize,strides,padding)

def vgg19(x,reuse=False,isTraining=True,convert=False):
    # opencv read in BGR not RGB
    VGG_MEAN = [103.939, 116.779, 123.68]
    x = tf.image.resize_images(x,(224,224))
    x = (x+1) * 127.5
    b = x[:,:,:,0] - VGG_MEAN[0]
    g = x[:,:,:,1] - VGG_MEAN[1]
    r = x[:,:,:,2] - VGG_MEAN[2]
    b = tf.reshape(b,[-1,224,224,1])
    g = tf.reshape(g,[-1,224,224,1])
    r = tf.reshape(r,[-1,224,224,1])

    x = tf.concat([b,g,r],axis=3)

    with tf.variable_scope("vgg19", reuse=reuse) as scope:
        if reuse: scope.reuse_variables()

        h = _conv_layer(x, 3, 64, 1, 3, "conv1_1", isTraining=isTraining)
        h = tf.nn.relu(h)
        h = _conv_layer(h, 64, 64, 1, 3, "conv1_2", isTraining=isTraining)
        h_1 = h
        h = tf.nn.relu(h)
        h = _max_pooling(h)

        h = _conv_layer(h, 64, 128, 1, 3, "conv2_1", isTraining=isTraining)
        h = tf.nn.relu(h)
        h = _conv_layer(h, 128, 128, 1, 3, "conv2_2", isTraining=isTraining)
        h_2 = h
        h = tf.nn.relu(h)
        h = _max_pooling(h)

        h = _conv_layer(h, 128, 256, 1, 3, "conv3_1", isTraining=isTraining)
        h = tf.nn.relu(h)
        h = _conv_layer(h, 256, 256, 1, 3, "conv3_2", isTraining=isTraining)
        h = tf.nn.relu(h)
        h = _conv_layer(h, 256, 256, 1, 3, "conv3_3", isTraining=isTraining)
        h = tf.nn.relu(h)
        h = _conv_layer(h, 256, 256, 1, 3, "conv3_4", isTraining=isTraining)
        h_3 = h
        h = tf.nn.relu(h)
        h = _max_pooling(h)

        h = _conv_layer(h, 256, 512, 1, 3, "conv4_1", isTraining=isTraining)
        h = tf.nn.relu(h)
        h = _conv_layer(h, 512, 512, 1, 3, "conv4_2", isTraining=isTraining)
        h = tf.nn.relu(h)
        h = _conv_layer(h, 512, 512, 1, 3, "conv4_3", isTraining=isTraining)
        h = tf.nn.relu(h)
        h = _conv_layer(h, 512, 512, 1, 3, "conv4_4", isTraining=isTraining)
        h_4 = h
        h = tf.nn.relu(h)
        h = _max_pooling(h)

        h = _conv_layer(h, 512, 512, 1, 3, "conv5_1", isTraining=isTraining)
        h = tf.nn.relu(h)
        h = _conv_layer(h, 512, 512, 1, 3, "conv5_2", isTraining=isTraining)
        h = tf.nn.relu(h)
        h = _conv_layer(h, 512, 512, 1, 3, "conv5_3", isTraining=isTraining)
        h = tf.nn.relu(h)
        h = _conv_layer(h, 512, 512, 1, 3, "conv5_4", isTraining=isTraining)
        h_5 = h
        h = tf.nn.relu(h)
        h = _max_pooling(h)

        if convert:
            h = _flatten(h)
            bs, f = h.get_shape().as_list()

            h = _fc_layer(h, f, 4096, "fc6")
            h = tf.nn.relu(h)
            h = _fc_layer(h, 4096, 4096, "fc7")
            h = tf.nn.relu(h)
            out = _fc_layer(h, 4096, 1000, "fc8")
        else:
            pass

    return h_1, h_2, h_3, h_4, h_5

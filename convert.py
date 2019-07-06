import tensorflow as tf
import numpy as np
from vgg import *
import os
SAVE_DIR = "modelvgg"

def printParam(scope):
    total_parameters = 0
    for variable in tf.trainable_variables(scope=scope):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("{} has {} parameters".format(scope, total_parameters))


if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

vggf = np.load("vgg19.npy", encoding='latin1',allow_pickle=True).item()
#print(type(vggf))
#print(len(vggf))
#print(vggf["conv1_1"][0].shape)
#print(sorted(vggf.keys()))
std = sorted(vggf.keys())

img_size=224
x = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
out = vgg19(x, convert=True)

#print(tf.trainable_variables(scope="vgg19"))
#print(len(tf.trainable_variables(scope="vgg19")))
printParam(scope="vgg19")
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for j in range(19):
    for i in range(2):
        update = tf.assign(tf.trainable_variables(scope="vgg19")[j*2+i],vggf[std[j]][i])
        up=sess.run(update)

var = sess.run(tf.trainable_variables(scope="vgg19")[1])
print(var)
print(vggf[std[0]][1])
print("done")
saver.save(sess,os.path.join(SAVE_DIR,"model.ckpt"),0)

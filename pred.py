import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from main import *
from model import *
DATASET_DIR = "data"
VAL_DIR ="val"
SAVE_DIR = "modelgan"
OUT_DIR = "outputgan"
UNLIMIT = False

def main(folder="test"):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    folder=folder
    files = os.listdir(folder)
    img_size = 360
    bs =8
    val_size =2

    start = time.time()

    x = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
    y =buildESRGAN_g(x,isTraining=False)


    g_vars = [x for x in tf.trainable_variables() if "ESRGAN_g" in x.name]
    print(len(tf.trainable_variables()))

    print("%.4e sec took building model"%(time.time()-start))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    #summary = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
    if ckpt: # checkpointがある場合
        #last_model = ckpt.all_model_checkpoint_paths[0]
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print ("load " + last_model)
        saver.restore(sess, last_model) # 変数データの読み込み
        print("succeed restore model")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    print("%.4e sec took initializing"%(time.time()-start))

    start = time.time()
    
    folder=folder
    files = os.listdir(folder)
    for i in range(len(files)):

        print(files[i])
        img = cv2.imread("{}/{}".format(folder,files[i]))
        img = (img-127.5)/127.5
        h,w = img.shape[:2]

        if UNLIMIT:
            if h > img_size or w > img_size:
                maxl = np.max((h, w))

                h = h*img_size//maxl
                w = w*img_size//maxl
                img =cv2.resize(img,(w,h))

        if h > img_size or w > img_size:
            print("height={} and weight={} both must be lower than {}".format(h,w,img_size))
            continue
        input = np.zeros((img_size,img_size,3))

        input[:h,:w]=img
        input= input.reshape(1,img_size,img_size,3)

        out = sess.run(y,feed_dict={x:input})

        X_ = cv2.resize(img,(w*4,h*4),interpolation = cv2.INTER_CUBIC)
        #X_ = cv2.resize(img,(w*4,h*4),interpolation = cv2.INTER_NEAREST)

        out = out.reshape(img_size*4,img_size*4,3)
        print(out.shape)
        Y_ = out[:h*4,:w*4]
        print("output shape is ",Y_.shape)

        X_ = (X_ + 1)*127.5
        Y_ = (Y_ + 1)*127.5
        cv2.imwrite("{}/{}_xval.png".format(OUT_DIR, i), X_)
        cv2.imwrite("{}/{}_yval.png".format(OUT_DIR, i), Y_)
        Z_ = np.concatenate((X_,Y_), axis=1)

        cv2.imwrite("{}/{}_val.png".format(OUT_DIR,i), Z_)

    print("%.4e sec took for predicting" %(time.time()-start))

if __name__ == '__main__':
    folder = "test"
    try:
        folder = sys.argv[1]
    except:
        pass
    main(folder)

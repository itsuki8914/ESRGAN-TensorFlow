import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt
from model import *
from btgen import BatchGenerator
from vgg import vgg19
TRAIN_LR_DIR = "train_lr"
TRAIN_HR_DIR = "train_hr"
VAL_LR_DIR = "val_lr"
VAL_HR_DIR = "val_hr"
VAL_DIR ="val"
TEST_DIR = "test"
SAVEPRE_DIR ="modelpre"
SAVE_DIR = "modelGAN"
SAVEIM_DIR ="sample"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if not os.path.exists(SAVEIM_DIR):
    os.makedirs(SAVEIM_DIR)

def tileImage(imgs):
    d = int(math.sqrt(imgs.shape[0]-1))+1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    r = np.zeros((h*d,w*d,3),dtype=np.float32)
    for idx,img in enumerate(imgs):
        idx_y = int(idx/d)
        idx_x = idx-idx_y*d
        r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
    return r

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

def foloderLength(folder):
    dir = folder
    paths = os.listdir(dir)
    return len(paths)

def main():
    img_size = 96
    bs = 4
    trans_lr = 1e-4

    start = time.time()

    batchgen = BatchGenerator(img_size=img_size,LRDir=TRAIN_LR_DIR,HRDir=TRAIN_HR_DIR,aug=True)
    valgen = BatchGenerator(img_size=img_size,LRDir=VAL_LR_DIR,HRDir=VAL_HR_DIR,aug=False)

    #save samples
    IN_ , OUT_ = batchgen.getBatch(4)[:4]
    print(IN_.shape)
    IN_ = tileImage(IN_)
    IN_ = cv2.resize(IN_,(img_size*2*4,img_size*2*4),interpolation = cv2.INTER_CUBIC)
    IN_ = (IN_ + 1)*127.5
    OUT_ = tileImage(OUT_)
    OUT_ = cv2.resize(OUT_,(img_size*4*2,img_size*4*2))
    OUT_ = (OUT_ + 1)*127.5
    Z_ = np.concatenate((IN_,OUT_), axis=1)
    cv2.imwrite("input.png",Z_)
    print("%s sec took sampling"%(time.time()-start))

    start = time.time()

    x = tf.placeholder(tf.float32, [bs, img_size, img_size, 3])
    t = tf.placeholder(tf.float32, [bs, img_size*4, img_size*4, 3])
    lr = tf.placeholder(tf.float32)

    y = buildESRGAN_g(x)
    test_y = buildESRGAN_g(x, reuse=True, isTraining=False)
    fake_y = buildESRGAN_d(y)
    real_y = buildESRGAN_d(t,reuse=True)

    vgg_y1, vgg_y2, vgg_y3, vgg_y4, vgg_y5 = vgg19(y)

    vgg_t1, vgg_t2, vgg_t3, vgg_t4, vgg_t5 = vgg19(t,reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=(real_y - tf.reduce_mean(fake_y)),labels=tf.ones_like(real_y)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=(fake_y - tf.reduce_mean(real_y)),labels=tf.zeros_like(fake_y)))

    g_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=(real_y - tf.reduce_mean(fake_y)),labels=tf.zeros_like(real_y))) * 1e-3
    g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=(fake_y - tf.reduce_mean(real_y)),labels=tf.ones_like(fake_y))) * 1e-3


    wd_g = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="Generator")
    wd_d = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope="Discriminator")

    wd_g = tf.reduce_sum(wd_g)
    wd_d = tf.reduce_sum(wd_d)

    L2_loss = tf.reduce_mean(tf.square(y - t))
    e_1 = tf.reduce_mean(tf.square(vgg_y1 - vgg_t1)) *2.8
    e_2 = tf.reduce_mean(tf.square(vgg_y2 - vgg_t2)) *0.2
    e_3 = tf.reduce_mean(tf.square(vgg_y3 - vgg_t3)) *0.08
    e_4 = tf.reduce_mean(tf.square(vgg_y4 - vgg_t4)) *0.2
    e_5 = tf.reduce_mean(tf.square(vgg_y5 - vgg_t5)) *75.0
    vgg_loss = (e_1 + e_2 + e_3 + e_4 + e_5) * 2e-7

    pre_loss = L2_loss + vgg_loss + wd_g
    g_loss = L2_loss + vgg_loss + g_loss_real + g_loss_fake + wd_g
    d_loss = d_loss_fake + d_loss_real + wd_d

    g_pre = tf.train.AdamOptimizer(1e-4,beta1=0.5).minimize(pre_loss, var_list=[
            x for x in tf.trainable_variables() if "ESRGAN_g"     in x.name])
    g_opt = tf.train.AdamOptimizer(lr,beta1=0.5).minimize(g_loss, var_list=[
            x for x in tf.trainable_variables() if "ESRGAN_g"     in x.name])
    d_opt = tf.train.AdamOptimizer(lr/2,beta1=0.5).minimize(d_loss, var_list=[
            x for x in tf.trainable_variables() if "ESRGAN_d" in x.name])

    print("%.4f sec took building"%(time.time()-start))
    printParam(scope="ESRGAN_g")
    printParam(scope="ESRGAN_d")
    printParam(scope="vgg19")

    g_vars = [x for x in tf.trainable_variables() if "ESRGAN_g"     in x.name]
    d_vars = [x for x in tf.trainable_variables() if "ESRGAN_d"     in x.name]
    vgg_vars = [x for x in tf.trainable_variables() if "vgg19"     in x.name]

    saver = tf.train.Saver()
    saver_vgg = tf.train.Saver(vgg_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(SAVEPRE_DIR)

    if ckpt: # is checkpoint exist
        last_model = ckpt.model_checkpoint_path
        #last_model = ckpt.all_model_checkpoint_paths[0]
        print ("load " + last_model)
        saver.restore(sess, last_model) # read variable data
        print("succeed restore model")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    ckpt_vgg = tf.train.get_checkpoint_state('modelvgg')
    last_model = ckpt_vgg.model_checkpoint_path
    saver_vgg.restore(sess, last_model)


    print("%.4e sec took initializing"%(time.time()-start))

    hist =[]
    hist_g =[]
    hist_d =[]

    start = time.time()
    print("start pretrain")
    for p in range(50001):
        batch_images_x, batch_images_t = batchgen.getBatch(bs)
        tmp, gen_loss,L2,vgg = sess.run([g_pre,pre_loss,L2_loss,vgg_loss], feed_dict={
            x: batch_images_x,
            t: batch_images_t
        })

        hist.append(gen_loss)
        print("in step %s, pre_loss =%.4e, L2_loss=%.4e, vgg_loss=%.4e" %(p, gen_loss, L2, vgg))

        if p % 100 == 0:
            batch_images_x, batch_images_t = valgen.getBatch(bs)

            out = sess.run(test_y,feed_dict={
                x:batch_images_x})
            X_ = tileImage(batch_images_x[:4])
            Y_ = tileImage(out[:4])
            Z_ = tileImage(batch_images_t[:4])

            X_ = cv2.resize(X_,(img_size*2*4,img_size*2*4),interpolation = cv2.INTER_CUBIC)

            X_ = (X_ + 1)*127.5
            Y_ = (Y_ + 1)*127.5
            Z_ = (Z_ + 1)*127.5
            Z_ = np.concatenate((X_,Y_,Z_), axis=1)

            cv2.imwrite("{0}/pre_{1:06d}.png".format(SAVEIM_DIR,int(p)),Z_)

            fig = plt.figure(figsize=(8,6), dpi=128)
            ax = fig.add_subplot(111)
            plt.title("Loss")
            plt.grid(which="both")
            plt.yscale("log")
            ax.plot(hist,label="gen_loss", linewidth = 0.25)
            plt.xlabel('step', fontsize = 16)
            plt.ylabel('loss', fontsize = 16)
            plt.legend(loc = 'upper right')
            plt.savefig("hist_pre.png")
            plt.close()

            print("%.4e sec took 100steps" %(time.time()-start))
            start = time.time()
        if p%5000==0 and p!=0:
            saver.save(sess,os.path.join(SAVEPRE_DIR,"model.ckpt"),p)


    print("start Discriminator")
    for d in range(0):
        batch_images_x, batch_images_t  = batchgen.getBatch(bs)

        tmp, dis_loss =sess.run([d_opt,d_loss,], feed_dict={
            x: batch_images_x,
            t: batch_images_t,
            lr:1e-4,
        })

        print("in step %s, dis_loss = %.4e"%(d, dis_loss))

    print("start GAN")
    for i in range(200001):
        batch_images_x, batch_images_t  = batchgen.getBatch(bs)

        tmp, gen_loss, L2, adv, vgg, = sess.run([g_opt,g_loss,L2_loss,g_loss_fake,vgg_loss], feed_dict={
            x: batch_images_x,
            t: batch_images_t,
            lr:trans_lr,
        })

        batch_images_x, batch_images_t  = batchgen.getBatch(bs)

        tmp, dis_loss =sess.run([d_opt,d_loss,], feed_dict={
            x: batch_images_x,
            t: batch_images_t,
            lr:trans_lr,
        })

        batch_images_x, batch_images_t  = batchgen.getBatch(bs)

        tmp, gen_loss, L2, adv, vgg, = sess.run([g_opt,g_loss,L2_loss,g_loss_fake,vgg_loss], feed_dict={
            x: batch_images_x,
            t: batch_images_t,
            lr:trans_lr,
        })

        if trans_lr > 1e-5:
            trans_lr = trans_lr * 0.99998

        print("in step %s, dis_loss = %.4e, gen_loss = %.4e"%(i, dis_loss, gen_loss))
        print("L2_loss=%.4e, adv_loss=%.4e, vgg_loss=%.4e"%(L2, adv, vgg))

        hist_g.append(gen_loss)
        hist_d.append(dis_loss)

        if i %100 ==0:
            batch_images_x, batch_images_t  = valgen.getBatch(bs)

            out = sess.run(test_y,feed_dict={
                x:batch_images_x})
            X_ = tileImage(batch_images_x[:4])
            Y_ = tileImage(out[:4])
            Z_ = tileImage(batch_images_t[:4])

            X_ = (X_ + 1)*127.5
            X_ = cv2.resize(X_,(img_size*4*2,img_size*4*2),interpolation = cv2.INTER_CUBIC)
            Y_ = (Y_ + 1)*127.5
            Z_ = (Z_ + 1)*127.5
            Z_ = np.concatenate((X_,Y_,Z_), axis=1)
            cv2.imwrite("{0}/gan_{1:06d}.png".format(SAVEIM_DIR,i),Z_)

            fig = plt.figure(figsize=(8,6), dpi=128)
            ax = fig.add_subplot(111)
            plt.title("Loss")
            plt.grid(which="both")
            plt.yscale("log")
            ax.plot(hist_g,label="gen_loss", linewidth = 0.25)
            ax.plot(hist_d,label="dis_loss", linewidth = 0.25)
            plt.xlabel('step', fontsize = 16)
            plt.ylabel('loss', fontsize = 16)
            plt.legend(loc = 'upper right')
            plt.savefig("hist.png")
            plt.close()

            print("%.4f sec took per 100steps, lr = %.4e" %(time.time()-start,trans_lr))
            start = time.time()


        if i%5000==0 and i !=0:
            saver.save(sess,os.path.join(SAVE_DIR,"model.ckpt"),i)


if __name__ == '__main__':
    main()

## We want to create a convolutional VAE that works based upon the GP-ConvVAE paper by
## Yoo et al. 2017. Some network architecture functions and cost are from their
## code, available at https://sites.google.com/view/yjyoo3312/
## As above, but to run the network just on the hand.
from __future__ import absolute_import, division, print_function
import timeit
#############
import os.path, sys

subfolder = 'DLC_info/'

sys.path.append(subfolder)

# Dependencies for video:
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy.misc import imread
from scipy.misc import imresize
import importlib
import imageio
imageio.plugins.ffmpeg.download()
from myconfig import *
import threading
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip

import time, glob
import pandas as pd
import numpy as np
import os
###############
import math

import glob

import prettytensor as pt

import tensorflow as tf

import pdb

from tensorflow.examples.tutorials.mnist import input_data

from deconv import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar
import shutil
from VAE_struct import *
from VAE_datagen import *
from VAE_costs import *
flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("nb_train",2000,"size of training set")
flags.DEFINE_integer("batch_size", 400, "batch size")
flags.DEFINE_integer('Imsizex',64,'Width to crop hand to')
flags.DEFINE_integer('Imsizey',64,'Height to crop hand to')
flags.DEFINE_integer('hsizex',256,'Width to rescale image to')
flags.DEFINE_integer('hsizey',256,'Height to rescale image to')
flags.DEFINE_integer('lag',0,'Number of frames to trail by for 3 frame reconstruction')
flags.DEFINE_integer('lead',0,'Number of frames to lead by for 3 frame reconstruction')
flags.DEFINE_integer("domain_size", (len(bodyparts)-4)*2, "dimension of the input domain")
flags.DEFINE_integer("updates_per_epoch", 200, "number of updates per epoch") #200
flags.DEFINE_integer("random_size", 32, "dimension of the random input")
flags.DEFINE_integer("max_epoch", 100, "max epoch") #100
flags.DEFINE_integer("hidden_size", 16, "size of the hidden VAE unit")
flags.DEFINE_float("jointnoise",0.01,"amount of noise to give on the joints")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_float("upweight_factor", 20000,"scale factor of peak height for upweighting (1 = standard gaussian)")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_string("traindirect","traindata_hand_null",'Name of directory where training data is stored')
flags.DEFINE_boolean("joints", True, "Joint information is not given")
flags.DEFINE_boolean("gen_samples", False, "Joint information is not given")
flags.DEFINE_boolean("efficient",True,'We reduce the number of layers by 1')


FLAGS = flags.FLAGS

train_foldername = 'imgs_prepost_hand_null'
# ltrain_foldername = 'limgs_h36m_act_all'
test_foldername = 'test_h36m_all_test4'

text_file = open("errors_vanilla_kl_1.txt", "w")

## Training routines:
if __name__ == "__main__":
    current_directory = os.getcwd()
    y_tensor_all,y_fullsize = get_data()

    ## One difference is that we now want to extract out frames around the hand,
    ## and designate that as our data. We should also take out all of the joint data
    ## for the paw (entries 1-4), and normalize to this.

    nframes=y_tensor_all.shape[0]
    os.chdir(current_directory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    ## Unfortunately, the video cannot be passed from a function without breaking
    ## the video's clip calling properties.

    clip = VideoFileClip(subfolder+Task+'/'+filename)
    ny,nx=clip.size #dimensions of frame (width, height)

    video=clip.crop(y1= -(ny/2)*scaley+shifty,y2=(ny/2)*scaley+shifty,x1 = -(nx/2)*scalex+shiftx,x2=nx/2*scalex+shiftx)

    # Now generate frames, and generate importance weightings
    if not os.path.exists(FLAGS.traindirect):
        print('Generating Training Frames')
        ## Generate training frames, and save the indices for later runs
        train_indices = generate_trainframes_multi_hand(y_tensor_all,y_fullsize,video)
        y_tensor_all = y_tensor_all[:,1:9]
        y_fullsize = y_fullsize[:,1:9]
        np.save(FLAGS.traindirect+'indices',train_indices)
        print('Done Generating Training Frames,Generating Weightings')

        current_y_train = y_tensor_all[train_indices,:]
        pre_y_train = y_tensor_all[train_indices-FLAGS.lag,:]
        post_y_train = y_tensor_all[train_indices+FLAGS.lead,:]
        y_train = np.concatenate((pre_y_train,current_y_train,post_y_train),axis = 1)
        print('Done Generating Training Frames,Generating Weightings')

        current_locations = y_fullsize[train_indices,:]
        pre_locations = y_fullsize[train_indices-FLAGS.lag,:]
        post_locations = y_fullsize[train_indices+FLAGS.lead,:]
        full_locations = np.concatenate((pre_locations,current_locations,post_locations),axis = 1)
        ## Generate importance filters to upweight the cost.
        filterfull = generate_filters_multi(full_locations)

    else:
        print('Training Data Already Generated')
        y_tensor_all = y_tensor_all[:,1:9]
        y_fullsize = y_fullsize[:,1:9]
        train_indices = np.load(FLAGS.traindirect+'indices.npy')
        current_y_train = y_tensor_all[train_indices,:]
        pre_y_train = y_tensor_all[train_indices-FLAGS.lag,:]
        post_y_train = y_tensor_all[train_indices+FLAGS.lead,:]
        y_train = np.concatenate((pre_y_train,current_y_train,post_y_train),axis = 1)
        print('Done Generating Training Frames,Generating Weightings')

        current_locations = y_fullsize[train_indices,:]
        pre_locations = y_fullsize[train_indices-FLAGS.lag,:]
        post_locations = y_fullsize[train_indices+FLAGS.lead,:]
        full_locations = np.concatenate((pre_locations,current_locations,post_locations),axis = 1)
        ## Generate importance filters to upweight the cost.
        filterfull = generate_filters_multi(full_locations)

        print('Done generating filters')

    ##################################
    # Set up loading of batches! We use a feed_dict to feed examples into a queue that is defined on placeholders. This gives us
    # flexibility when we want to re-run the network later.
    if FLAGS.joints:
        names = ['input','noise','filters','joints']
        shapes = [[None,3*FLAGS.Imsizex*FLAGS.Imsizey*3],[None,FLAGS.hidden_size+3*FLAGS.domain_size+2],[None,3*FLAGS.Imsizex*FLAGS.Imsizey*3],[None,3*FLAGS.domain_size]]
        out,enqueue_op,placeholders = setup_preload(names,shapes)

        ## These will take values from the queue if not specified, but can be fed to as standard otherwise.
        input_tensor = tf.placeholder_with_default(out['input'], shape=shapes[0],name = 'in_def')
        y_tensor = tf.placeholder_with_default(out['joints'], shape=shapes[3],name = 'y_def')
        e_tensor = tf.placeholder_with_default(out['noise'], shape=shapes[1],name = 'e_def')
        f_tensor = tf.placeholder_with_default(out['filters'], shape=shapes[2],name ='f_def')
    else:

        names = ['input','noise','filters']
        shapes = [[FLAGS.batch_size,3*FLAGS.Imsizex*FLAGS.Imsizey*3],[None,FLAGS.hidden_size+3*FLAGS.domain_size+2],[None,3*FLAGS.Imsizex*FLAGS.Imsizey*3]]
        out,enqueue_op,placeholders = setup_preload(names,shapes)

        ## These will take values from the queue if not specified, but can be fed to as standard otherwise.
        input_tensor = tf.placeholder_with_default(out['input'], shape=shapes[0],name = 'in_def')
        e_tensor = tf.placeholder_with_default(out['noise'], shape=shapes[1],name = 'e_def')
        f_tensor = tf.placeholder_with_default(out['filters'], shape=shapes[2],name = 'f_def')

    # Now design the architecture of the network.
    ##################################
    if FLAGS.joints:
        # for train
        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            with pt.defaults_scope(phase=pt.Phase.train):
                with tf.variable_scope("model_g") as scope:
                    # The encoder networks:
                    # For the hidden state:
                    hidden_state = tf.placeholder_with_default(encoder_eff_multi_small(input_tensor), shape=[FLAGS.batch_size,2 * FLAGS.hidden_size + 2],name = 'hidden_input')
                    # For the joints:
                    # joint_state = tf.placeholder_with_default(encoder_joints_multi_small(input_tensor), shape=[FLAGS.batch_size,3*2*FLAGS.domain_size],name = 'joint_parameters')
                    tensor_e, mean, stddev = decoder_eff_multi_small(hidden_state, y_tensor, e_tensor)

            with pt.defaults_scope(phase=pt.Phase.test):
                with tf.variable_scope("model_g", reuse=True) as scope:
                    hidden_state = tf.placeholder_with_default(encoder_eff_multi_small(input_tensor), shape=[FLAGS.batch_size,2 * FLAGS.hidden_size + 2],name = 'hidden_input')
                    # joint_state = tf.placeholder_with_default(encoder_joints_multi(input_tensor), shape=[FLAGS.batch_size,3*2*FLAGS.domain_size],name = 'joint_parameters')
                    sampled_tensor, _, _ = decoder_eff_multi_small(hidden_state, y_tensor, e_tensor)

            with pt.defaults_scope(phase=pt.Phase.test):
                with tf.variable_scope("model_g", reuse=True) as scope:
                    hidden_state = tf.placeholder_with_default(encoder_eff_multi_small(input_tensor), shape=[FLAGS.batch_size,2 * FLAGS.hidden_size + 2],name = 'hidden_input')
                    # joint_state = tf.placeholder_with_default(encoder_joints_multi_small(input_tensor), shape=[FLAGS.batch_size,3*2*FLAGS.domain_size],name = 'joint_parameters')
                    test_tensor, _, _ = decoder_eff_multi_small(hidden_state, y_tensor, e_tensor)
    else:

        with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            with pt.defaults_scope(phase=pt.Phase.train):

                with tf.variable_scope("model_g") as scope:
                    hidden_state = tf.placeholder_with_default(encoder_eff_multi(input_tensor), shape=[FLAGS.batch_size,2 * FLAGS.hidden_size + 2],name = 'hidden_input')
                    joint_state = tf.placeholder_with_default(encoder_joints_multi(input_tensor), shape=[FLAGS.batch_size,2*FLAGS.domain_size],name = 'joint_parameters')
                    #### THIS IS NOT IMPLEMENTED
                    tensor_e, mean, stddev = decoder_nj_eff(hidden_state,e_tensor)

            with pt.defaults_scope(phase=pt.Phase.test):
                with tf.variable_scope("model_g", reuse=True) as scope:
                    hidden_state = tf.placeholder_with_default(encoder_eff(input_tensor), shape=[FLAGS.batch_size,2 * FLAGS.hidden_size + 2],name = 'hidden_input')
                    joint_state = tf.placeholder_with_default(encoder_joints(input_tensor), shape=[FLAGS.batch_size,2*FLAGS.domain_size],name = 'joint_parameters')
                    sampled_tensor, _, _ = decoder_nj_eff(hidden_state,e_tensor)


            with pt.defaults_scope(phase=pt.Phase.test):
                with tf.variable_scope("model_g", reuse=True) as scope:
                    hidden_state = tf.placeholder_with_default(encoder_eff(input_tensor), shape=[FLAGS.batch_size,2 * FLAGS.hidden_size + 2],name = 'hidden_input')
                    joint_state = tf.placeholder_with_default(encoder_joints(input_tensor), shape=[FLAGS.batch_size,2*FLAGS.domain_size],name = 'joint_parameters')
                    test_tensor, _, _ = decoder_nj_eff(hidden_state,e_tensor)

    # Define costs on the outputs of these networks.
    ##################################

    # This is the generic KL cost for the variational autoencoder.
    vae_loss = get_vae_cost(mean, stddev)  # do not consider mean and stddev generated from GP.

    # Reconstruction loss, as found by monte carlo samples of the network.
    ## For now, we only train on the central frame.
    central_frame = input_tensor[:,FLAGS.Imsizex*FLAGS.Imsizey*3:2*FLAGS.Imsizex*FLAGS.Imsizey*3]
    central_filter = f_tensor[:,FLAGS.Imsizex*FLAGS.Imsizey*3:2*FLAGS.Imsizex*FLAGS.Imsizey*3]
    rec_loss = get_filtered_cost(tensor_e, central_frame, central_filter, FLAGS.Imsizex,FLAGS.Imsizey)
    alpha = 0.1*FLAGS.nb_train
    if FLAGS.joints:
        # The "Extra Term" loss, found by monte carlo estimates of a "cross entropy" term.
        # central_joint_state = joint_state[:,2*FLAGS.domain_size:4*FLAGS.domain_size]
        # VI_prior_loss = get_VIprior_cost(central_joint_state,y_tensor[:,FLAGS.domain_size:2*FLAGS.domain_size])
        loss = vae_loss+rec_loss#-alpha*VI_prior_loss

    iter_epoch = FLAGS.max_epoch

    iter_epoch = FLAGS.max_epoch

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate,epsilon=1.0).minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=5)


    with tf.Session() as sess:
        sess.run(init)
        ##############
        # Start preloading!
        ##############
        coord,thread = start_preload_multi(sess,enqueue_op,y_train,filterfull,placeholders)

        joint_status = ['no_joints','joints']
        for epoch in range(iter_epoch):
            training = 0.0

            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(maxval=FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            nframes=y_tensor_all.shape[0]

            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                # Where the action happens.
                ###########################################
                if FLAGS.joints:
                    _, costeval = sess.run([optimizer,loss])

                else:
                    _, costeval = sess.run([optimizer,loss])

                # print(np.sum(np.log(distcosts),axis=1))
                ###########################################
                training += costeval

            training = training / \
                         (FLAGS.updates_per_epoch * FLAGS.batch_size)

            print("Loss %f" % (
            training))
            text_file.write(
                "%f\n" % (training))


            imgs_folder = os.path.join(FLAGS.working_directory, train_foldername)

            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)

            save_path = saver.save(sess, os.path.join(imgs_folder, 'epoch' + str(epoch) + Task + filename + date + joint_status[int(FLAGS.joints)] + 'queuemodel.ckpt'))
            ## We can generate samples from a completed epoch to see how we're doing.
            if FLAGS.gen_samples:
                if FLAGS.joints:
                    rec_imgs = sess.run([sampled_tensor])
                else:
                    rec_imgs = sess.run([sampled_tensor])

                print([i.shape for i in rec_imgs])
                for k in range(10):
                    img = rec_imgs[0][k,:,:,:]
                    one = 3 * k
                    two = 3 * k + 1
                    three = 3 * k + 2

                    img = img * 255.0
                    img_set = img.reshape([FLAGS.Imsizex, FLAGS.Imsizey, 3])

                    imsave(os.path.join(imgs_folder, 'epochqueue'+str(epoch) + Task + filename + date + joint_status[int(FLAGS.joints)]+'%03d.png') % two,
                           img_set[0:FLAGS.Imsizex, 0:FLAGS.Imsizey, 0:3])

            else:
                pass

    text_file.close()
    sess.close()
    coord.request_stop()
    coord.join([thread])

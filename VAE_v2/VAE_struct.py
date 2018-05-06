## All functions having to do with network structure of the VAE goes here.
## There are two different encoders that differ in the number of layers that they use, and
## Four different decoders, that are conjunctions of numbers of layers and inclusion of
## joint information.
import sys
import prettytensor as pt
import tensorflow as tf
subfolder = 'DLC_info/'

sys.path.append(subfolder)
from myconfig import *

flags = tf.flags
FLAGS = flags.FLAGS

def encoder_fullsize(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    input_img_tensor = input_tensor[:, :FLAGS.Imsizex * FLAGS.Imsizey * 3]
    return (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride = 2).
            conv2d(5, 256, edges = 'VALID').
            conv2d(5, 512, stride = 2).
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.hidden_size + 2,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

def encoder_eff(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    input_img_tensor = input_tensor[:, :FLAGS.Imsizex * FLAGS.Imsizey * 3]
    return (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride = 4).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.hidden_size + 2,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

def encoder_eff_multi(input_tensor):
    '''Create encoder network, with three different branches.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    ## Unshape the inputs again:
    preinput_img_tensor = input_tensor[:, :FLAGS.Imsizex * FLAGS.Imsizey * 3]
    input_img_tensor = input_tensor[:, FLAGS.Imsizex * FLAGS.Imsizey * 3:2*FLAGS.Imsizex * FLAGS.Imsizey * 3]
    postinput_img_tensor = input_tensor[:, 2*FLAGS.Imsizex * FLAGS.Imsizey * 3:]

    pre =   (pt.wrap(preinput_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride = 4).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten())
            # fully_connected(2 * FLAGS.hidden_size + 2,
            #                 activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

    post =  (pt.wrap(postinput_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride = 4).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten())

    return (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride = 4).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            join(others = (pre,post)).
            fully_connected(2 * FLAGS.hidden_size + 2,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

def encoder_eff_multi_med(input_tensor):
    '''Create encoder network, with three different branches.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    ## Unshape the inputs again:
    preinput_img_tensor = input_tensor[:, :FLAGS.Imsizex * FLAGS.Imsizey * 3]
    input_img_tensor = input_tensor[:, FLAGS.Imsizex * FLAGS.Imsizey * 3:2*FLAGS.Imsizex * FLAGS.Imsizey * 3]
    postinput_img_tensor = input_tensor[:, 2*FLAGS.Imsizex * FLAGS.Imsizey * 3:]

    pre =   (pt.wrap(preinput_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten())
            # fully_connected(2 * FLAGS.hidden_size + 2,
            #                 activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

    post =  (pt.wrap(postinput_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten())

    return (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            join(others = (pre,post)).
            fully_connected(2 * FLAGS.hidden_size + 2,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

def encoder_eff_multi_small(input_tensor):
    '''Create encoder network, with three different branches.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    ## Unshape the inputs again:
    preinput_img_tensor = input_tensor[:, :FLAGS.Imsizex * FLAGS.Imsizey * 3]
    input_img_tensor = input_tensor[:, FLAGS.Imsizex * FLAGS.Imsizey * 3:2*FLAGS.Imsizex * FLAGS.Imsizey * 3]
    postinput_img_tensor = input_tensor[:, 2*FLAGS.Imsizex * FLAGS.Imsizey * 3:]

    pre =   (pt.wrap(preinput_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten())
            # fully_connected(2 * FLAGS.hidden_size + 2,
            #                 activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

    post =  (pt.wrap(postinput_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten())

    return (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            join(others = (pre,post)).
            fully_connected(2 * FLAGS.hidden_size + 2,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

def encoder_joints(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    input_img_tensor = input_tensor[:, :FLAGS.Imsizex * FLAGS.Imsizey * 3]
    return (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride = 4).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.domain_size,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

def encoder_joints_multi(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    input_img_tensor = input_tensor[:, :FLAGS.Imsizex * FLAGS.Imsizey * 3]
    pre =  (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride = 4).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.domain_size,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior
    main = (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride = 4).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.domain_size,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

    post = (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride = 4).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.domain_size,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior
    return tf.concat((pre,main,post),axis = 1)

def encoder_joints_multi_small(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    input_img_tensor = input_tensor[:, :FLAGS.Imsizex * FLAGS.Imsizey * 3]
    pre =  (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.domain_size,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior
    main = (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.domain_size,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

    post = (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.domain_size,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior
    return tf.concat((pre,main,post),axis = 1)

def encoder_joints_multi_small(input_tensor):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28*3]

    Returns:
        A tensor that expresses the encoder network
    '''
    input_img_tensor = input_tensor[:, :FLAGS.Imsizex * FLAGS.Imsizey * 3]
    pre =  (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.domain_size,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior
    main = (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.domain_size,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior

    post = (pt.wrap(input_img_tensor).
            reshape([None, FLAGS.Imsizex, FLAGS.Imsizey, 3]).
            conv2d(5, 16, stride=2).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride=2).
            conv2d(5, 256, edges = 'VALID').
            dropout(0.9).
            flatten().
            fully_connected(2 * FLAGS.domain_size,
                            activation_fn=None,name = 'hidden')).tensor  # one for gaussian prior, the other for axis prior
    return tf.concat((pre,main,post),axis = 1)


def decoder_eff(input_tensor=None, y_tensor= None, e_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode, half number of the batch
        y_tensor: domain of the vectors, full number of batch
        e_tensor is a noise vector.
        # Not used: dist_tensor is the set of supervised y tensors (?)
    Returns:
        A tensor that expresses the decoder network: return batch number of reconstructed image
        half of them is a result of original encoded vector
        the other half is a generation w.r.t gaussian regression # Regressing from pose space onto image space.
        # We should cut out this second part for now, and just do regular of vae.
    '''
    # epsilon_c = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size]) #random parameters for co-domain
    # epsilon_d = tf.random_normal([FLAGS.batch_size, FLAGS.domain_size])

    epsilon_c = e_tensor[:, :FLAGS.hidden_size]  # random parameters for co-domain
    epsilon_d = e_tensor[:, FLAGS.hidden_size:FLAGS.hidden_size+FLAGS.domain_size]  # random parameters for domain



    mean = tf.cast(input_tensor[:, :FLAGS.hidden_size],dtype = tf.float32)
    stddev = tf.cast(tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:2 * FLAGS.hidden_size])),dtype = tf.float32)
    latent_sample = mean + tf.cast(epsilon_c,dtype = tf.float32) * stddev

    # What is below describes the process of using the joint locations to improve regression.
    mean_y = y_tensor[:, :FLAGS.domain_size]



    # print(mean_y,epsilon_d)
    domain_sample =  mean_y+tf.cast(FLAGS.jointnoise * epsilon_d,dtype = tf.float32)

    # final sample P(z|y)
    input_sample = tf.concat([latent_sample, domain_sample], 1)

    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size+FLAGS.domain_size]).
            deconv2d(4, 256, edges='VALID').
            deconv2d(5, 128, stride = 4).
            deconv2d(5, 64, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(5, 16, stride=2).
            deconv2d(5, 3, stride=2, activation_fn=tf.nn.sigmoid)
            ).tensor, mean, stddev  # last 1 channel means foreground region

def decoder_eff_multi(input_tensor=None, y_tensor= None, e_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode, half number of the batch
        y_tensor: domain of the vectors, full number of batch
        e_tensor is a noise vector.
        # Not used: dist_tensor is the set of supervised y tensors (?)
    Returns:
        A tensor that expresses the decoder network: return batch number of reconstructed image
        half of them is a result of original encoded vector
        the other half is a generation w.r.t gaussian regression # Regressing from pose space onto image space.
        # We should cut out this second part for now, and just do regular of vae.
    '''
    # epsilon_c = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size]) #random parameters for co-domain
    # epsilon_d = tf.random_normal([FLAGS.batch_size, FLAGS.domain_size])
    single_size = FLAGS.hidden_size+FLAGS.domain_size
    epsilon_c = e_tensor[:, :FLAGS.hidden_size]  # random parameters for co-domain
    pre_epsilon_d = e_tensor[:, FLAGS.hidden_size:single_size]  # random parameters for domain
    epsilon_d = e_tensor[:, single_size:single_size+FLAGS.domain_size]
    post_epsilon_d = e_tensor[:, single_size+FLAGS.domain_size:single_size+2*FLAGS.domain_size]

    mean = tf.cast(input_tensor[:, :FLAGS.hidden_size],dtype = tf.float32)
    stddev = tf.cast(tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:2 * FLAGS.hidden_size])),dtype = tf.float32)
    latent_sample = mean + tf.cast(epsilon_c,dtype = tf.float32) * stddev

    # What is below describes the process of using the joint locations to improve regression.
    mean_y = y_tensor[:, :FLAGS.domain_size]
    pre_mean_y = y_tensor[:, FLAGS.domain_size:2*FLAGS.domain_size]
    post_mean_y = y_tensor[:, 2*FLAGS.domain_size:3*FLAGS.domain_size]

    # print(mean_y,epsilon_d)
    domain_sample =  mean_y+tf.cast(FLAGS.jointnoise * epsilon_d,dtype = tf.float32)
    pre_domain_sample =  pre_mean_y+tf.cast(FLAGS.jointnoise * pre_epsilon_d,dtype = tf.float32)
    post_domain_sample =  post_mean_y+tf.cast(FLAGS.jointnoise * post_epsilon_d,dtype = tf.float32)

    # final sample P(z|y)
    input_sample = tf.concat([latent_sample, pre_domain_sample,domain_sample,post_domain_sample], 1)

    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size+3*FLAGS.domain_size]).
            deconv2d(4, 256, edges='VALID').
            deconv2d(5, 128, stride = 4).
            deconv2d(5, 64, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(5, 16, stride=2).
            deconv2d(5, 3, stride=2, activation_fn=tf.nn.sigmoid)
            ).tensor, mean, stddev  # last 1 channel means foreground region

def decoder_eff_multi_small(input_tensor=None, y_tensor= None, e_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode, half number of the batch
        y_tensor: domain of the vectors, full number of batch
        e_tensor is a noise vector.
        # Not used: dist_tensor is the set of supervised y tensors (?)
    Returns:
        A tensor that expresses the decoder network: return batch number of reconstructed image
        half of them is a result of original encoded vector
        the other half is a generation w.r.t gaussian regression # Regressing from pose space onto image space.
        # We should cut out this second part for now, and just do regular of vae.
    '''
    # epsilon_c = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size]) #random parameters for co-domain
    # epsilon_d = tf.random_normal([FLAGS.batch_size, FLAGS.domain_size])
    single_size = FLAGS.hidden_size+FLAGS.domain_size
    epsilon_c = e_tensor[:, :FLAGS.hidden_size]  # random parameters for co-domain
    pre_epsilon_d = e_tensor[:, FLAGS.hidden_size:single_size]  # random parameters for domain
    epsilon_d = e_tensor[:, single_size:single_size+FLAGS.domain_size]
    post_epsilon_d = e_tensor[:, single_size+FLAGS.domain_size:single_size+2*FLAGS.domain_size]

    mean = tf.cast(input_tensor[:, :FLAGS.hidden_size],dtype = tf.float32)
    stddev = tf.cast(tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:2 * FLAGS.hidden_size])),dtype = tf.float32)
    latent_sample = mean + tf.cast(epsilon_c,dtype = tf.float32) * stddev

    # What is below describes the process of using the joint locations to improve regression.
    mean_y = y_tensor[:, :FLAGS.domain_size]
    pre_mean_y = y_tensor[:, FLAGS.domain_size:2*FLAGS.domain_size]
    post_mean_y = y_tensor[:, 2*FLAGS.domain_size:3*FLAGS.domain_size]

    # print(mean_y,epsilon_d)
    domain_sample =  mean_y+tf.cast(FLAGS.jointnoise * epsilon_d,dtype = tf.float32)
    pre_domain_sample =  pre_mean_y+tf.cast(FLAGS.jointnoise * pre_epsilon_d,dtype = tf.float32)
    post_domain_sample =  post_mean_y+tf.cast(FLAGS.jointnoise * post_epsilon_d,dtype = tf.float32)

    # final sample P(z|y)
    input_sample = tf.concat([latent_sample, pre_domain_sample,domain_sample,post_domain_sample], 1)

    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size+3*FLAGS.domain_size]).
            deconv2d(4, 256, edges='VALID').
            deconv2d(5, 64, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(5, 16, stride=2).
            deconv2d(5, 3, stride=2, activation_fn=tf.nn.sigmoid)
            ).tensor, mean, stddev  # last 1 channel means foreground region

def decoder_eff_multi_med(input_tensor=None, y_tensor= None, e_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode, half number of the batch
        y_tensor: domain of the vectors, full number of batch
        e_tensor is a noise vector.
        # Not used: dist_tensor is the set of supervised y tensors (?)
    Returns:
        A tensor that expresses the decoder network: return batch number of reconstructed image
        half of them is a result of original encoded vector
        the other half is a generation w.r.t gaussian regression # Regressing from pose space onto image space.
        # We should cut out this second part for now, and just do regular of vae.
    '''
    # epsilon_c = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size]) #random parameters for co-domain
    # epsilon_d = tf.random_normal([FLAGS.batch_size, FLAGS.domain_size])
    single_size = FLAGS.hidden_size+FLAGS.domain_size
    epsilon_c = e_tensor[:, :FLAGS.hidden_size]  # random parameters for co-domain
    pre_epsilon_d = e_tensor[:, FLAGS.hidden_size:single_size]  # random parameters for domain
    epsilon_d = e_tensor[:, single_size:single_size+FLAGS.domain_size]
    post_epsilon_d = e_tensor[:, single_size+FLAGS.domain_size:single_size+2*FLAGS.domain_size]

    mean = tf.cast(input_tensor[:, :FLAGS.hidden_size],dtype = tf.float32)
    stddev = tf.cast(tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:2 * FLAGS.hidden_size])),dtype = tf.float32)
    latent_sample = mean + tf.cast(epsilon_c,dtype = tf.float32) * stddev

    # What is below describes the process of using the joint locations to improve regression.
    mean_y = y_tensor[:, :FLAGS.domain_size]
    pre_mean_y = y_tensor[:, FLAGS.domain_size:2*FLAGS.domain_size]
    post_mean_y = y_tensor[:, 2*FLAGS.domain_size:3*FLAGS.domain_size]

    # print(mean_y,epsilon_d)
    domain_sample =  mean_y+tf.cast(FLAGS.jointnoise * epsilon_d,dtype = tf.float32)
    pre_domain_sample =  pre_mean_y+tf.cast(FLAGS.jointnoise * pre_epsilon_d,dtype = tf.float32)
    post_domain_sample =  post_mean_y+tf.cast(FLAGS.jointnoise * post_epsilon_d,dtype = tf.float32)

    # final sample P(z|y)
    input_sample = tf.concat([latent_sample, pre_domain_sample,domain_sample,post_domain_sample], 1)

    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size+3*FLAGS.domain_size]).
            deconv2d(4, 256, edges='VALID').
            deconv2d(4, 128, stride = 2).
            deconv2d(5, 64, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(5, 16, stride=2).
            deconv2d(5, 3, stride=2, activation_fn=tf.nn.sigmoid)
            ).tensor, mean, stddev  # last 1 channel means foreground region



def decoder_nj_eff(input_tensor=None, e_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode, half number of the batch
        y_tensor: domain of the vectors, full number of batch
        e_tensor is a noise vector.
        # Not used: dist_tensor is the set of supervised y tensors (?)
    Returns:
        A tensor that expresses the decoder network: return batch number of reconstructed image
        half of them is a result of original encoded vector
        the other half is a generation w.r.t gaussian regression # Regressing from pose space onto image space.
        # We should cut out this second part for now, and just do regular of vae.
    '''
    # epsilon_c = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size]) #random parameters for co-domain
    # epsilon_d = tf.random_normal([FLAGS.batch_size, FLAGS.domain_size])

    epsilon_c = e_tensor[:, :FLAGS.hidden_size]  # random parameters for co-domain

    mean = input_tensor[:, :FLAGS.hidden_size]
    stddev = tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:2 * FLAGS.hidden_size]))
    latent_sample = mean + tf.cast(epsilon_c,tf.float32) * stddev

    # # What is below describes the process of using the joint locations to improve regression.
    # mean_y = y_tensor[:, :FLAGS.domain_size]
    # domain_sample = mean_y + 0.5 * epsilon_d

    # final sample P(z|y)
    # input_sample = tf.concat([latent_sample, domain_sample], 1)

    return (pt.wrap(latent_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
            deconv2d(4, 256, edges='VALID').
            deconv2d(5, 128, stride = 4).
            deconv2d(5, 64, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(5, 16, stride=2).
            deconv2d(5, 3, stride=2, activation_fn=tf.nn.sigmoid)
            ).tensor, mean, stddev  # last 1 channel means foreground region

def decoder_fullsize(input_tensor=None, y_tensor= None, e_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode, half number of the batch
        y_tensor: domain of the vectors, full number of batch
        e_tensor is a noise vector.
        # Not used: dist_tensor is the set of supervised y tensors (?)
    Returns:
        A tensor that expresses the decoder network: return batch number of reconstructed image
        half of them is a result of original encoded vector
        the other half is a generation w.r.t gaussian regression # Regressing from pose space onto image space.
        # We should cut out this second part for now, and just do regular of vae.
    '''
    # epsilon_c = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size]) #random parameters for co-domain
    # epsilon_d = tf.random_normal([FLAGS.batch_size, FLAGS.domain_size])

    epsilon_c = e_tensor[:, :FLAGS.hidden_size]  # random parameters for co-domain
    epsilon_d = e_tensor[:, FLAGS.hidden_size:FLAGS.hidden_size+FLAGS.domain_size]  # random parameters for domain

    mean = tf.cast(input_tensor[:, :FLAGS.hidden_size],dtype = tf.float32)
    stddev = tf.cast(tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:2 * FLAGS.hidden_size])),dtype = tf.float32)
    latent_sample = mean + tf.cast(epsilon_c,dtype = tf.float32) * stddev

    # What is below describes the process of using the joint locations to improve regression.
    mean_y = y_tensor[:, :FLAGS.domain_size]
    # print(mean_y,epsilon_d)
    domain_sample = mean_y + tf.cast(FLAGS.jointnoise * epsilon_d,tf.float32)

    # final sample P(z|y)
    input_sample = tf.concat([latent_sample, domain_sample], 1)

    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size+FLAGS.domain_size]).
            deconv2d(4, 512, edges='VALID').
            deconv2d(5, 256, stride = 2).
            deconv2d(5, 128, stride = 2).
            deconv2d(5, 64, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(5, 16, stride=2).
            deconv2d(5, 3, stride=2, activation_fn=tf.nn.sigmoid)
            ).tensor, mean, stddev  # last 1 channel means foreground region

def decoder_nj_fullsize(input_tensor=None, e_tensor=None):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode, half number of the batch
        y_tensor: domain of the vectors, full number of batch
        e_tensor is a noise vector.
        # Not used: dist_tensor is the set of supervised y tensors (?)
    Returns:
        A tensor that expresses the decoder network: return batch number of reconstructed image
        half of them is a result of original encoded vector
        the other half is a generation w.r.t gaussian regression # Regressing from pose space onto image space.
        # We should cut out this second part for now, and just do regular of vae.
    '''
    # epsilon_c = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size]) #random parameters for co-domain
    # epsilon_d = tf.random_normal([FLAGS.batch_size, FLAGS.domain_size])

    epsilon_c = e_tensor[:, :FLAGS.hidden_size]  # random parameters for co-domain

    mean = input_tensor[:, :FLAGS.hidden_size]
    stddev = tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:2 * FLAGS.hidden_size]))
    latent_sample = mean + tf.cast(epsilon_c,tf.float32) * stddev

    # # What is below describes the process of using the joint locations to improve regression.
    # mean_y = y_tensor[:, :FLAGS.domain_size]
    # domain_sample = mean_y + 0.5 * epsilon_d

    # final sample P(z|y)
    # input_sample = tf.concat([latent_sample, domain_sample], 1)

    return (pt.wrap(latent_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
            deconv2d(4, 512, edges='VALID').
            deconv2d(5, 256, edges='VALID').
            deconv2d(5, 128, stride = 2).
            deconv2d(5, 64, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(5, 16, stride=2).
            deconv2d(5, 3, stride=2, activation_fn=tf.nn.sigmoid)
            ).tensor, mean, stddev  # last 1 channel means foreground region

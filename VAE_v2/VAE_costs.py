import sys
import prettytensor as pt
import tensorflow as tf
import threading
subfolder = 'DLC_info/'

sys.path.append(subfolder)
from myconfig import *
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.misc import imread
from scipy.ndimage.filters import gaussian_filter

flags = tf.flags
FLAGS = flags.FLAGS
# flags.DEFINE_integer("hidden_size", 16, "size of the hidden VAE unit")
# flags.DEFINE_integer("domain_size", len(bodyparts)*2, "dimension of the input domain")
# flags.DEFINE_integer("batch_size", 400, "batch size")



def generate_filters(full_locations,framelocs):
    for imind in range(FLAGS.nb_train):
        for joint in range(len(bodyparts)):

            jointloc = np.round(full_locations[imind,2*joint:2*joint+2])
            jointloc= jointloc.astype(int)
            if np.all(jointloc<[FLAGS.Imsizex,FLAGS.Imsizey]):
                framelocs[imind,jointloc[1],jointloc[0]] = 1

    filters = np.stack([gaussian_filter(framelocs[i,:,:], 10,truncate = 2,mode = "constant")*FLAGS.upweight_factor+1 for i in range(FLAGS.nb_train)],axis = 0)
    filters = filters.reshape(FLAGS.nb_train,FLAGS.Imsizex,FLAGS.Imsizey,1)

    filterfull = np.tile(filters,(1,1,1,3))
    filterfull = filterfull.reshape(FLAGS.nb_train,FLAGS.Imsizex*FLAGS.Imsizey*3)
    return filterfull

## Where all cost-function related functions are stored, relating to the VAE for
## compression.
def get_filtered_cost(output_tensor, target_tensor, filters, im_sizex,im_sizey, epsilon=1e-8):

    output_img_all = tf.reshape(output_tensor, [FLAGS.batch_size, FLAGS.Imsizex, FLAGS.Imsizey, 3])
    output_img_rgb = output_img_all[:, :, :, :3]
    outout_tensor_rgb = tf.reshape(output_img_rgb, [FLAGS.batch_size, FLAGS.Imsizex * FLAGS.Imsizey * 3])

    target_img_tensor = target_tensor[:, :FLAGS.Imsizex * FLAGS.Imsizey * 3]
    # Note that you've not included this normlaization here, menaing reconstruction is highly prioritized... 1/(FLAGS.L+FLAGS.M)*
    return tf.reduce_sum(((-target_img_tensor * tf.log(outout_tensor_rgb + epsilon) -
                         (1.0 - target_img_tensor) * tf.log(1.0 - outout_tensor_rgb + epsilon))*filters))

def get_reconstruction_cost(output_tensor, target_tensor, im_sizex,im_sizey, epsilon=1e-8):
    '''Reconstruction loss

    Cross entropy reconstruction loss

    Args:
        output_tensor:Imsizey produces by decoder
        target_tensor: the target tensor that we want to reconstruct
        epsilon:
    '''

    output_img_all = tf.reshape(output_tensor, [FLAGS.batch_size, FLAGS.Imsizex, FLAGS.Imsizey, 3])
    output_img_rgb = output_img_all[:, :, :, :3]
    outout_tensor_rgb = tf.reshape(output_img_rgb, [FLAGS.batch_size, FLAGS.Imsizex * FLAGS.Imsizey * 3])

    target_img_tensor = target_tensor[:, :FLAGS.Imsizex * FLAGS.Imsizey * 3]
    # Note that you've not included this normlaization here, menaing reconstruction is highly prioritized... 1/(FLAGS.L+FLAGS.M)*
    return tf.reduce_sum((-target_img_tensor * tf.log(outout_tensor_rgb + epsilon) -
                         (1.0 - target_img_tensor) * tf.log(1.0 - outout_tensor_rgb + epsilon)))

def get_vae_cost(mean, stddev, epsilon=1e-8):

    '''VAE loss
        See the paper


    Args:
        mean:
        stddev:
        epsilon:
    '''
    return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                                2.0 * tf.log(stddev + epsilon) - 1.0))

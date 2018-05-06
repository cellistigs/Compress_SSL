### A tool to take trained networks, and then generate videos from them using
### temporary folders.
import os.path, sys
from collections import deque
#subfolder=os.getcwd().split('Evaluation-Tools')[0]
# subfolder = '../../../Motion_Code_To_Standardize/DeepMarkerlessTracking_beta_taiga_edits'
subfolder = 'DLC_info'
sys.path.append(subfolder)
# add parent directory: (where nnet & config are!)
# sys.path.append(subfolder+"/pose-tensorflow/")
# sys.path.append(subfolder+"/Generating_a_Training_Set")
# Dependencies for video:
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.misc import imread
import importlib
import imageio
imageio.plugins.ffmpeg.download()
from myconfig import *
from skimage import io
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import skvideo.io
import subprocess
from skimage import io
import skimage
import skimage.color
from scipy.misc import imresize
import time, glob
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import prettytensor as pt
# from ConvVAE_feed_queue_select import encoder_fullsize,decoder_fullsize,encoder_eff,decoder_eff,decoder_nj_eff,decoder_nj_fullsize,get_data,FLAGS
from VAE_struct import *
from VAE_datagen import *
from VAE_costs import *
flags = tf.flags
# FLAGS = flags.FLAGS
from ConvVAE_hand_multi import FLAGS
# The pipeline is as follows:

## 1. construct the network architecture
# fullsize = 1
# References to model:
basefolder = 'imgs_prepost_hand_null/'
# basefolder = './'
joint_status = ['no_joints','joints']
print(joint_status[int(FLAGS.joints)])
basename = 'epoch' + str(FLAGS.max_epoch-4) + Task + filename + date + joint_status[int(FLAGS.joints)]+'queuemodel.ckpt'

# basename = 'epoch74' + Task + filename + date + joint_status[int(FLAGS.joints)]+'queuemodel.ckpt'
# if fullsize == 1:
# basefolder = 'imgs_h36m_act_all/'
metadata = basename+'.meta'#'epoch98costa_feeddictmodel.ckpt.meta'
data = basename
# if fullsize == 0:
#     basefolder = 'imgs_h36m_act_all/'
#     metadata = '/epoch99costa_joints_effmodel.ckpt.meta'
#     data = '/epoch99costa_joints_effmodel.ckpt'

# # Initialize architecture:
# input_tensor = tf.placeholder(tf.float32, [None, Imsize * Imsize * 3], name="in")
# y_tensor = tf.placeholder(tf.float32, [None, FLAGS.domain_size], name="y")
# e_tensor = tf.placeholder(tf.float32, [None, FLAGS.hidden_size+FLAGS.domain_size+2], name="e")
# print(FLAGS.hidden_size)
# with pt.defaults_scope(activation_fn=tf.nn.elu,
#                        batch_normalize=True,
#                        learned_moments_update_rate=0.0003,
#                        variance_epsilon=0.001,
#                        scale_after_normalization=True):
#
#     with pt.defaults_scope(phase=pt.Phase.test):
#         with tf.variable_scope("model_g", reuse=False) as scope:
#             sampled_tensor, _, _ = decoder_fullsize(encoder_fullsize(input_tensor), y_tensor, e_tensor)

# pixelwise = np.mean((reconstruction - sampled_tensor)**2)
## 2. Link to the video file
# video = subfolder+'/videos'+'/'+Task+'/'+filename
# clip = VideoFileClip(video)
# ny,nx=clip.size #dimensions of frame (width, height)
#
# clip=clip.crop(y1= -(ny/2)*scaley+shifty,y2=(ny/2)*scaley+shifty,x1 = -(nx/2)*scalex+shiftx,x2=nx/2*scalex+shiftx)
#
# print("Duration of video [s], ", clip.duration, "fps, ", clip.fps, "Cropped frame dimensions: ", clip.size)
#


# # Extract relevant index sets for the train and test sets
# trained,untrained = traintest_split(imgfolder,nframes)

## 3. Restore Weights, pass each frame through the network, reformat, and calculate the residual.
## Weights:

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
saver = tf.train.import_meta_graph(basefolder+metadata)
saver.restore(sess,basefolder+data)

# coord = tf.train.Coordinator() # To coordinate child threads before they
# # Start running.
# ## We now need to start these threads:
# threads = tf.train.start_queue_runners(coord=coord,sess=sess)

# graph = tf.get_default_graph()
# print([n.name for n in tf.get_default_graph().as_graph_def().node])
graph = tf.get_default_graph()
# sampled_tensor = graph.get_tensor_by_name("model_g_1/deconv2d_5/Sigmoid:0") # for fullsize
sampled_tensor = graph.get_tensor_by_name("model_g_1/deconv2d_4/Sigmoid:0")
hidden_activations = graph.get_tensor_by_name('model_g_1/hidden/MatMul:0')
# hidden_activations = graph.get_tensor_by_name('model_g_1/hidden/model_g_1/hidden/add/activations:0')
 # Depends on how many layers you're using at the moment.
input_tensor = graph.get_tensor_by_name('in_def:0')
y_tensor = graph.get_tensor_by_name('y_def:0')
e_tensor = graph.get_tensor_by_name('e_def:0')
## Restore joint locations and video:
current_directory = os.getcwd()
y_tensor_all,y_fullsize = get_data()
batch = 1

y_tensor_all = y_tensor_all.astype(float)

## Now we use y_tensor_all to do some preprocessing.
mv_diff = np.diff(y_tensor_all,axis = 0)[:,1:]
dists = np.linalg.norm(mv_diff,axis = 1)
plt.plot(dists)
plt.savefig('Diff_dist.png')
movement = np.where(dists>0.1)[0]
nframes=y_tensor_all.shape[0]
os.chdir(current_directory)
clip = VideoFileClip(subfolder+'/'+Task+'/'+filename)
# subtask = 'shaved_reaching_aux'
# filename = 'shaved_mark.avi'
# clip = VideoFileClip(subfolder+'/videos/'+Task+'/'+subtask+"/"+filename)

ny,nx=clip.size #dimensions of frame (width, height)

clip=clip.crop(y1= -(ny/2)*scaley+shifty,y2=(ny/2)*scaley+shifty,x1 = -(nx/2)*scalex+shiftx,x2=nx/2*scalex+shiftx)
# print(clip.get_frame((batch*400)*1./clip.fps)-clip.get_frame((batch*400+5000)*1./clip.fps))
direct = 'temp'+Task+joint_status[int(FLAGS.joints)]+'inv_sample_prepost_hand_null'
if not os.path.isdir(direct):
    os.mkdir(direct)

## First find activations:
from sklearn.decomposition import PCA

batches = int(nframes/FLAGS.batch_size)
print(batches)
PCA_done = 0
randomize = 0

if randomize == 1:
    randbatch = np.random.permutation(batches)

else:
    randbatch = np.arange(batches)

vidindex = 0
xaxis = deque(maxlen = FLAGS.batch_size/2)
residuals = deque(maxlen = FLAGS.batch_size/2)
print('here!')
hidden = []

null = 1
if null == 0:
    lead = 1
    llead = 2
else:
    lead = 0
    llead = 0

for basebatch in range(batches):
    batch = basebatch
    rand_ind = randbatch[basebatch]
    rand_ind = basebatch
    imgs_batch = np.zeros((FLAGS.batch_size,3*FLAGS.Imsizex*FLAGS.Imsizey*3))
    for index in range(FLAGS.batch_size):
        image=img_as_ubyte(clip.get_frame((batch*FLAGS.batch_size+index)*1./clip.fps))
        # Preprocessing:
        img_resize = imresize(image,[FLAGS.hsizex,FLAGS.hsizey])
        ## now normalize to 0-1 range:
        img_normalized = img_resize/255.
        ## Flatten:
        # input_frame = img_normalized.reshape([1,FLAGS.Imsizex*FLAGS.Imsizey*3])

        input_n = img_as_ubyte(clip.get_frame((batch*FLAGS.batch_size+index+lead)*1./clip.fps))
        resize_n = imresize(input_n,[FLAGS.hsizex,FLAGS.hsizey])
        normalized_n = resize_n/255.

        input_nn = img_as_ubyte(clip.get_frame((batch*FLAGS.batch_size+index+llead)*1./clip.fps))
        resize_nn = imresize(input_nn,[FLAGS.hsizex,FLAGS.hsizey])
        normalized_nn = resize_nn/255.

        # Now find the hand:
        x_eff = (y_fullsize[batch*FLAGS.batch_size+index+lead,2]).astype(int)
        y_eff = (y_fullsize[batch*FLAGS.batch_size+index+lead,3]).astype(int)
        prex_eff = (y_fullsize[batch*FLAGS.batch_size+index,2]).astype(int)
        prey_eff = (y_fullsize[batch*FLAGS.batch_size+index,3]).astype(int)
        postx_eff = (y_fullsize[batch*FLAGS.batch_size+index+llead,2]).astype(int)
        posty_eff = (y_fullsize[batch*FLAGS.batch_size+index+llead,3]).astype(int)

        prehand = cropframe(prex_eff,prey_eff,img_normalized)
        hand = cropframe(x_eff,y_eff,normalized_n)
        posthand = cropframe(postx_eff,posty_eff,normalized_nn)

        full = np.concatenate((prehand,hand,posthand))
        ## Run the network:
        imgs_batch[index,:] = full.reshape([1,3*FLAGS.Imsizex*FLAGS.Imsizey*3])

        # Find the right y tensors as well:
        y_current = y_tensor_all[FLAGS.batch_size*rand_ind:FLAGS.batch_size*rand_ind+FLAGS.batch_size,1:9]
        y_n = y_tensor_all[FLAGS.batch_size*rand_ind+lead:FLAGS.batch_size*rand_ind+lead+FLAGS.batch_size,1:9]
        y_nn = y_tensor_all[FLAGS.batch_size*rand_ind+llead:FLAGS.batch_size*rand_ind+llead+FLAGS.batch_size,1:9]
        # print(y_current.shape,y_pre.shape,y_post.shape)
        y_batch = np.concatenate((y_current,y_n,y_nn),axis = 1)
    # noise = np.random.randn(FLAGS.batch_size,FLAGS.hidden_size+FLAGS.domain_size+2)
    noise = np.zeros((FLAGS.batch_size,FLAGS.hidden_size+3*FLAGS.domain_size+2))
    if FLAGS.joints:
        out = sess.run([hidden_activations,sampled_tensor],feed_dict={input_tensor:imgs_batch,y_tensor:y_batch,e_tensor:noise})
        all_activations = out[0]
        reconstruction = out[1]
    else:
        out = sess.run([hidden_activations,sampled_tensor],feed_dict={input_tensor:imgs_batch,e_tensor:noise})
        all_activations = out[0]
        reconstruction = out[1]
    img_reconstruct = reconstruction.reshape([FLAGS.batch_size,FLAGS.Imsizex,FLAGS.Imsizey,3])
    mean_activations = all_activations[:,:FLAGS.hidden_size]
    hidden.append(mean_activations)
        # residual = img_resize-img_reconstruct

    # print(img_orig)
    print('now here!')

    colorscheme=['r','g','y','b','m','r','g','y','b'] #colors for those bodyparts.

    reconstruct = np.arange(int(FLAGS.batch_size/2))
    for z in range(FLAGS.batch_size):
        if (z+basebatch*400 or z-1+basebatch*400 or z+1+basebatch*400) in movement:
            f = plt.figure(figsize=(20,10))
            gs = gridspec.GridSpec(2,3,width_ratios=[1,1,1],height_ratios=[3,1])
            axlist = []
            ax00 = plt.subplot(gs[0,0])
            ax01 = plt.subplot(gs[0,1])
            ax02 = plt.subplot(gs[0,2])
            ax10 = plt.subplot(gs[1,0])
            ax11 = plt.subplot(gs[1,1])
            ax12 = plt.subplot(gs[1,2])

            # f,axarr = plt.subplots(2,3,figsize=(20,10))
            residual = img_reconstruct[z,:,:,:]-imgs_batch[z,FLAGS.Imsizex*FLAGS.Imsizey*3:2*FLAGS.Imsizex*FLAGS.Imsizey*3].reshape(FLAGS.Imsizex,FLAGS.Imsizey,3)
            xshift = y_fullsize[FLAGS.batch_size*basebatch+z,2]
            yshift = y_fullsize[FLAGS.batch_size*basebatch+z,3]
            ax00.imshow(residual)
            ax00.set_title('Residual')
            ax01.imshow(imgs_batch[z,FLAGS.Imsizex*FLAGS.Imsizey*3:2*FLAGS.Imsizex*FLAGS.Imsizey*3].reshape(FLAGS.Imsizex,FLAGS.Imsizey,3))
            for part in [1,2,3,4]:
                ax01.scatter(y_fullsize[FLAGS.batch_size*basebatch+z,2*part]-xshift+FLAGS.Imsizex/2,y_fullsize[FLAGS.batch_size*basebatch+z,2*part+1]-yshift+FLAGS.Imsizey/2,color =colorscheme[part],alpha = 0.5)
            ax01.set_title('Original')
            ax02.imshow(img_reconstruct[z,:,:,:])

            ax02.set_title('Reconstruction')
            rms = np.sqrt(np.mean(residual**2))
            residuals.append(rms)
            if len(xaxis)<FLAGS.batch_size/2:
                xaxis.appendleft(-vidindex)
            # x_eff = (fullsize[FLAGS.batch_size*basebatch+z,2]).astype(int)
            # y_eff = (fullsize[FLAGS.batch_size*basebatch+z,3]).astype(int)
            # xmin = np.max(((x_eff-FLAGS.Imsizex/10).astype(int),0))
            # xmax = np.min(((x_eff+FLAGS.Imsizex/10).astype(int),FLAGS.Imsizex))
            # ymin = np.max(((y_eff-FLAGS.Imsizey/10).astype(int),0))
            # ymax = np.min(((y_eff+FLAGS.Imsizey/10).astype(int),FLAGS.Imsizey))
            # img_orig = imgs_batch[z,FLAGS.Imsizex*FLAGS.Imsizey*3:2*FLAGS.Imsizex*FLAGS.Imsizey*3].reshape(FLAGS.Imsizex,FLAGS.Imsizey,3)
            # ax10.imshow(np.concatenate((img_reconstruct[z,ymin:ymax,xmin:xmax,:],img_orig[ymin:ymax,xmin:xmax,:])))
            ax11.plot(xaxis,residuals,'o')
            # axarr[1,1].plot(data_reduced[FLAGS.batch_size*basebatch+z,0],data_reduced[FLAGS.batch_size*basebatch+z,1],'ro',markersize = 7)
            # axarr[1,1].plot(data_reduced[FLAGS.batch_size*basebatch+zprev,0],data_reduced[FLAGS.batch_size*basebatch+zprev,1],'o',color = 'black',markersize=6)
            # axarr[1,1].plot(data_reduced[FLAGS.batch_size*basebatch+zpprev,0],data_reduced[FLAGS.batch_size*basebatch+zpprev,1],'o',color = 'black',markersize = 4)
            # axarr[1,1].plot(data_reduced[FLAGS.batch_size*basebatch+zppprev,0],data_reduced[FLAGS.batch_size*basebatch+zppprev,1],'o',color = 'black',markersize = 2)
            # axarr[1,1].plot(data_reduced[FLAGS.batch_size*basebatch+zpppprev,0],data_reduced[FLAGS.batch_size*basebatch+zpppprev,1],'o',color = 'black',markersize = 1)
            # axarr[1,1].plot(data_reduced[FLAGS.batch_size*basebatch+zppppprev,0],data_reduced[FLAGS.batch_size*basebatch+zppppprev,1],'o',color = 'black',markersize = 1)
            # ax10.set_title('Zoom Paw Reconstruction/True')
            ax11.set_xlim([-200,200])
            ax11.set_ylim([0,0.2])
            ax12.plot(mean_activations[:,0::5])
            ax12.axvline(x = z,color = 'black')
            ax12.set_title('Example Hidden Layer Activities')
            ax12.set_xlabel('Frames')
            ax12.set_ylabel('Units')
            ax11.set_title('Residual Evolution')
            ax11.set_xlabel('Relative Frame Number')
            ax11.set_ylabel('Residual Error (RMSE)')
            print(np.mean(residual))
            ax00.axis('off')
            ax01.axis('off')
            ax02.axis('off')
            ax10.axis('off')
            plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
            plt.savefig(direct+'/arbitrary'+'%05d.png'%int(vidindex))
            plt.close('all')
            # zppppprev = zpppprev
            # zpppprev = zppprev
            # zppprev = zpprev
            # zpprev = zprev
            # zprev = z
            vidindex+=1
# Iterate through frames:



# good_batches = [0]



os.chdir(direct)
print("Generating video")
subprocess.call(['ffmpeg', '-framerate', str(60/5), '-i', 'arbitrary%05d.png', '-r', '30','Video_invsample'+'.mp4'])
for file_name in glob.glob("*.png"):
    os.remove('./'+file_name)
os.chdir("../")




## 5. save this file to a temporary folder

## 6. use ffmpeg to turn all of these into a file.

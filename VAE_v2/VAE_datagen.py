## All functions that have to do with generation of data, and enqueuing of said
## data for increased efficiency at runtimeself.
import sys
import os
import pandas as pd
import prettytensor as pt
import tensorflow as tf
import threading
subfolder = 'DLC_info/'

sys.path.append(subfolder)
from myconfig import *
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.misc import imread,imresize,imsave
from sklearn.neighbors.kde import KernelDensity

flags = tf.flags
FLAGS = flags.FLAGS
# flags.DEFINE_integer("hidden_size", 16, "size of the hidden VAE unit")
# flags.DEFINE_integer("domain_size", len(bodyparts)*2, "dimension of the input domain")
# flags.DEFINE_integer("batch_size", 400, "batch size")


def generate_trainframes(y_tensor_all,video):
    ## Establish probability-scaled sampling:
    X = y_tensor_all
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    scores = kde.score_samples(X)
    normalized = np.round(1/scores/np.min(1/scores)).astype(int)
    lengthvec = np.arange(np.shape(X)[0])
    to_sample = np.repeat(lengthvec,normalized)

    os.makedirs(FLAGS.traindirect)
    # Select a number of training frames:

    train_indices= np.random.choice(to_sample,FLAGS.nb_train,replace = False)
    y_train = y_tensor_all[train_indices,:]
    for i,train_index in enumerate(train_indices):
        trainframe = video.get_frame((train_index)*1./video.fps)
        resized = imresize(trainframe,[FLAGS.Imsizex,FLAGS.Imsizey])
        imsave(os.path.join(FLAGS.traindirect, 'trainframe'+'%05d.png') % i,
               resized)
    return train_indices

def get_data():
    shuffle = Shuffles[shuffleindex]

    # Name for scorer:
    scorer='DeepCut'+"_resnet"+str(resnet)+"_"+Task+str(date)+'shuffle'+str(shuffle)+'_'+str(trainingsiterations)

    ##################################################
    # Datafolder
    ##################################################



    ##################################################
    # Plotting indi
    ##################################################

    # basefolder = '../../../Motion_Code_To_Standardize/DeepMarkerlessTracking_beta_dev/videos/'
    # basefolder = '../DeepMarkerlessTracking_beta/videos/'
    # basefolder='../../videos/' #where your folder with videos is.
    # Task = 'shaved_mark'
    folder = Task+'/'
    # Task = 'shaved_mark'
    # filename = 'shaved_mark.avi'

    os.chdir(subfolder+folder)

    # First analyze the video you have as your training example
    videos=np.sort([fn for fn in os.listdir(os.curdir) if ("avi" in fn)])

    for video in videos:
        clip = VideoFileClip(video)
        ny,nx=clip.size #dimensions of frame (width, height)

        clip=clip.crop(y1= -(ny/2)*scaley+shifty,y2=(ny/2)*scaley+shifty,x1 = -(nx/2)*scalex+shiftx,x2=nx/2*scalex+shiftx)

        dataname=video.split('.')[0]+scorer+'.h5'
        Dataframe=pd.read_hdf(dataname)
        data = Dataframe[scorer].as_matrix()

        ## Extract Frames and dimensions
        joints = int(round(data.shape[1]/3))
        frames = data.shape[0]

        ## Reformat as is appropriate for the variational autoencoder:
        ## Extract x and y separately
        ## First resize to the image:
        vidscalex = FLAGS.Imsizex/(nx*scalex)
        vidscaley = FLAGS.Imsizey/(ny*scaley)
        # print(vidscalex*nx*scalex,vidscaley*ny*scaley)

        datax = data[:,0::3]
        datay = data[:,1::3]
        # print(np.max(datax),np.max(datay),nx*scalex,ny*scaley)
        fullsize = np.zeros((frames,joints*2))
        fullsize[:,0::2] = datax*vidscalex
        fullsize[:,1::2] = datay*vidscaley
        ## Recenter relative to one coordinate:
        rdatax = datax-datax[:,2:3].repeat(joints,axis = 1)
        rdatay = datay-datay[:,2:3].repeat(joints,axis = 1)

        viddata = np.zeros((frames,joints*2))
        viddata[:,0::2] = rdatax
        viddata[:,1::2] = rdatay

        viddata = viddata/100+0.3 ## One thing that seems important is that
        # the other dataset, when processed, generated information that was entirely with a positive mean, and with
        # approximately 90% positive weights

    return viddata,fullsize


def generate_train_batch_feeddict(y_train,filterfull):


    cnt = 0

    source_list = []
    joint_list = []
    index_list = []
    for idx in range(0, 1):  # for each video

        ## Select out the appropriate frames of the video:
        vid_list = np.zeros((FLAGS.batch_size,FLAGS.Imsizex*FLAGS.Imsizey*3))
        joint_list = np.zeros((FLAGS.batch_size,FLAGS.domain_size))
        noise_list = np.random.randn(FLAGS.batch_size,FLAGS.hidden_size+FLAGS.domain_size+2)
        # times = []
        offset = np.random.randint(0,10)
        filter_list = filterfull[offset*FLAGS.batch_size:offset*FLAGS.batch_size+FLAGS.batch_size]
        for i in range(FLAGS.batch_size):
            index = i+offset*FLAGS.batch_size
            filename = os.path.join(FLAGS.traindirect, 'trainframe'+'%05d.png'%index)
            img = imread(filename)
            # lmt = imresize(imread(jpg_set), [2*Imsize, 2*Imsize])
            img_normalized = img / 255.0
            # # start = timeit.timeit()
            # permed = i
            # # print(permed)
            # # print(permed)
            # img = video.get_frame(permed*1./video.fps)
            # img_resize = imresize(img,[Imsizex,Imsizey])
            # # end = timeit.timeit()
            # # elapsed = end-start
            #     ## now normalize to 0-1 range:
            # img_normalized = img_resize/255.
            #     ## Flatten:
            #     # input_frame = img_normalized.reshape([1,Imsize*Imsize*3])
            joints = y_train[index:index+1,:]

            vid_list[i,:]= img_normalized.reshape(FLAGS.Imsizex*FLAGS.Imsizey*3)
            joint_list[i:i+1,:]= joints
            # times.append(elapsed)
            # print(np.mean(times),'mean time elapsed to get resize')
        # img_batch = np.stack(vid_list,axis = 0)
        # joint_batch = np.vstack(joint_list)


        return vid_list,joint_list,noise_list,filter_list

### Define the architecture that lets you feed training examples through the input queue using placeholders
def setup_preload(names,shapes):
    QUEUE_SIZE = 20
    placeholders = {name:tf.placeholder(tf.float32,shape=shape,name = name) for (name,shape) in zip(names,shapes)}
    placeholders_list = list(placeholders.values())


    q = tf.FIFOQueue(QUEUE_SIZE,[tf.float32]*len(names))
    enqueue_op = q.enqueue(placeholders_list)
    batch_list = q.dequeue()
    out = {}
    for i,name in enumerate(names):
        out[name] = batch_list[i]
        out[name].set_shape(shapes[i])
    return out,enqueue_op,placeholders

### The loading location that now takes outputs of the batch generation, and feeds them to the queue.
def load_enqueue(sess,enqueue_op,coord,y_train,filterfull,placeholders):
    if FLAGS.joints:
        while not coord.should_stop():
            ## Generate batches
            vidbatch,jointbatch,noisebatch,filterbatch = generate_train_batch_feeddict(y_train,filterfull)

            batchdict = {'input':vidbatch,'noise':noisebatch,'filters':filterbatch,'joints':jointbatch}

            food = {placeholder:batchdict[name] for (name,placeholder) in placeholders.items()}

            sess.run(enqueue_op,feed_dict=food)
    else:
        while not coord.should_stop():
            ## Generate batches
            vidbatch,jointbatch,noisebatch,filterbatch = generate_train_batch_feeddict(y_train,filterfull)

            batchdict = {'input':vidbatch,'noise':noisebatch,'filters':filterbatch}

            food = {placeholder:batchdict[name] for (name,placeholder) in placeholders.items()}

            sess.run(enqueue_op,feed_dict=food)


def start_preload(sess,enqueue_op,y_train,filterfull,placeholders):
    coord = tf.train.Coordinator()
    t = threading.Thread(target = load_enqueue,
                         args = (sess,enqueue_op,coord,y_train,filterfull,placeholders))

    t.start()
    return coord,t

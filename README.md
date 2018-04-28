# Compress_SSL
Repository for rotation projects with Liam Paninski/John Cunningham. 

Dependencies: 
Python 2.7
Standard Scientific Computing Packages: (Scipy, Numpy, Matplotlib, Pandas)
scikit-learn
scikit-image
scikit-video
tensorflow-gpu (1.6)
prettytensor
progressbar
imageio
moviepy

It's recommended to install the above in a separate anaconda virtual environment. 

This package contains code that compresses video data (when used with joint information derived from DeepLabCut, https://arxiv.org/abs/1804.03142). The main body of training code can be called by running:

$$ python ConvVAE_feed_queue_select.py

This will save a trained model in a folder specified by the variable "train_foldername", in the script mentioned above. 

Then, analysis can be done by:
1. Generating a video corresponding to the reconstruction, via 
$$ python video_maker_queue.py
and
2. Exploring the latent space via
$$ python noise_explorer.py



# coding: utf-8

# In[ ]:

# myconfig.py:

#Step 1:
Task='reach_full'
# Filename and path to behavioral video:
filename='Mouse1vid1.avi'

#Step 2:
bodyparts=["Tongue","F1","F2","F3","Wrist","Joystick"] #Exact sequence of labels as were put by annotator in *.csv file
Scorers=['Taiga'] #who is labeling?

#Step 3:
date='Mar24'
scorer='Taiga'

# Portion of the video to sample from in step 1. Set to 1 by default.
portion = 0.7
sample_offset = 3000 # The number of frames to skip at the beginning of a video, 
# Userparameters for training set. Other parameters can be set in pose_cfg.yaml
Shuffles=[1]            # Ids for shuffles, i.e. range(5) for 5 shuffles
TrainingFraction=[0.95]  # Fraction of labeled images used for training
shuffleindex = -1

invisibleboundary=10
# Which resnet to use (these are parameters reflected in the pose_cfg.yaml file)
resnet=50
trainingsiterations='1030000'

## Finetuning parameters:
nb_frames = 50 ## number of frames to use for finetuning, per bodypart.

# For evaluation
# To evaluate model that was trained most set this to: "-1"
# To evaluate all models (training stages) set this to: "all"
snapshotindex=-1

# This file will be written to here from Step1, giving the shift indices for the
# file. Later, we can specify the scale too.

shiftx = 326.0

shifty = 340.0

scalex = 0.6

scaley = 0.5

shiftx = 326.0

shifty = 340.0

scalex = 0.6

scaley = 0.5

shiftx = 326.0

shifty = 220.0

scalex = 0.6

scaley = 0.5

shiftx = 326.0

shifty = 220.0

scalex = 0.6

scaley = 0.4

shiftx = 326.0

shifty = 220.0

scalex = 0.4

scaley = 0.4

shiftx = 326.0

shifty = 220.0

scalex = 0.375

scaley = 0.4

shiftx = 326.0

shifty = 220.0

scalex = 0.625

scaley = 0.4

shiftx = 326.0

shifty = 220.0

scalex = 0.5

scaley = 0.4

shiftx = 326.0

shifty = 270.0

scalex = 0.5

scaley = 0.4

shiftx = 326.0

shifty = 270.0

scalex = 0.5625

scaley = 0.5

shiftx = 326.0

shifty = 270.0

scalex = 0.5625

scaley = 0.5

shiftx = 326.0

shifty = 270.0

scalex = 0.5625

scaley = 0.5

shiftx = 326.0

shifty = 270.0

scalex = 0.5625

scaley = 0.5

shiftx = 326.0

shifty = 270.0

scalex = 0.5625

scaley = 0.5

shiftx = 326.0

shifty = 270.0

scalex = 0.5625

scaley = 0.5

shiftx = 326.0

shifty = 270.0

scalex = 0.5625

scaley = 0.5

shiftx = 326.0

shifty = 270.0

scalex = 0.5625

scaley = 0.5

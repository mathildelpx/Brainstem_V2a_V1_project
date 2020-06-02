import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
import time
import pickle
import imageio
from utils.import_data import *
from utils.functions_calcium_analysis import *


fishlabel = '190104_F2'
trial = '8'
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'

bouts = load_bout_object(output_path, fishlabel, trial)
exp = load_experiment(output_path, fishlabel)[0]
struct= load_output_struct(fishlabel, trial, output_path+'dataset/', '/network/lustre/iss01/wyart/analyses/2pehaviour/suite2p_output/')
TA = struct['TA_all']

tif_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/suite2p_output/190104_F2/8/suite2p/plane0/reg_tif/'

# loading the tif image
im1 = Image.open(tif_path + 'file000_chan0.tif')
im2 = Image.open(tif_path + 'file001_chan0.tif')

# converting the tiff image in array
imarray1 = np.array(im1)
imarray2 = np.array(im2)

nFrames = 846

# the tiff array will be in 3D: 2D of number of pixels, and the third D is the time.
tiff_array = np.zeros((imarray1.shape[0], imarray1.shape[1], nFrames))

# get each frame of the tif file to be concatenated to the array
for i, frame in enumerate(ImageSequence.Iterator(im1)):
    tiff_array[:, :, i] = frame

for j, frame in enumerate(ImageSequence.Iterator(im2)):
    tiff_array[:, :, i + j] = frame

# get baseline
baseline_array = tiff_array[:,:,1:20]

baseline_value = np.zeros(tiff_array.shape)

for i in range(tiff_array.shape[0]):
    for j in range(tiff_array.shape[1]):
        value = np.median(baseline_array[i,j,:])
        if value == 0:
            value = 0.0001
        baseline_value[i, j, :] = value

dff_array = ((tiff_array-baseline_value)/baseline_value)*1000

# save dff array as an image
dff_video = []
for array in range(846):
    frame = Image.fromarray(dff_array[:,:,array].astype(np.uint8))
    dff_video.append(frame)

imageio.mimsave('test_dff_190104_F2_8.avi', dff_video, format='avi')

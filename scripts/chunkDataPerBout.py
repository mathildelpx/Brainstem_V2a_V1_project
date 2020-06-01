from PIL import Image, ImageSequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from utils.tiff_managment import cut_calcium_imaging, cut_behavior, resize_condense_behavior
from utils.import_data import load_config_file, load_trials_correspondence


fishlabel = '190104_F2'
trial = '7'
data_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/'
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/fig/'
fps2P = 200
fpsBehavior = 4.2

try:
    os.mkdir(output_path + fishlabel + '/' + trial + '/' + 'single_bout_vizu')
except FileExistsError:
    print('Fish and trial already analyzed')
    pass

# add a step with a macro in ImageJ to convert the tiff in DFF tiff

# Load analyzed data
# output of pipeline
df_bouts = pd.read_pickle('/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/dataset/' + fishlabel + '/' + fishlabel + '_analyzed_bout_dataset_' + trial)
for bout in df_bouts.index:
    try:
        os.mkdir(output_path + fishlabel + '/' + trial + '/' + 'single_bout_vizu/' + str(bout))
    except FileExistsError:
        pass


op = output_path + fishlabel + '/' + trial + '/' + 'single_bout_vizu/'

analysis_log = pd.DataFrame({'Start_CI_frame': [np.nan]*len(df_bouts.index),
                             'End_CI_frame': [np.nan]*len(df_bouts.index),
                             'Start_CI_time': [np.nan]*len(df_bouts.index),
                             'End_CI_time': [np.nan]*len(df_bouts.index),
                             'Start_beh_frame': [np.nan]*len(df_bouts.index),
                             'End_beh_frame': [np.nan]*len(df_bouts.index),
                             'Start_beh_time': [np.nan]*len(df_bouts.index),
                             'End_beh_time': [np.nan]*len(df_bouts.index)}, index=df_bouts.index)

## Calcium imaging
files = os.listdir(data_path+'suite2p_output/'+fishlabel+'/'+trial+'/')
for file in files:
    if file.endswith('.tiff'):
        tif_file = file
reg_tiff = Image.open(data_path+'suite2p_output/'+fishlabel+'/'+trial+'/'+tif_file)

# select subset of tiff images
pd.Series(df_bouts.index).apply(cut_calcium_imaging, args=(df_bouts, reg_tiff, fps2P, fpsBehavior, op, analysis_log))

analysis_log.to_csv(op+'vizu_log.csv')

plt.figure()
plt.title('Time correspondance for behavior and ci gif')
plt.plot(analysis_log['Start_CI_time'], 'o', label='ci start time')
plt.plot(analysis_log['Start_beh_time'], 'o', label='start beh time')
plt.legend()
plt.tight_layout()
plt.xlabel('bout')
plt.ylabel('Time to start gif')
plt.savefig(op+'time_correspondance_gif.png')





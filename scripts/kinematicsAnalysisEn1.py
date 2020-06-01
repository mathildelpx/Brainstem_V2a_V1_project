import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.get_kinematics import get_frames_time_points, tau_calculation
from utils.import_data import load_output_struct, load_suite2p_outputs, load_config_file

# Initialization: write here file name, path, frequency of acquisition and list of eventulations.
fishlabel = '190207_F1'
trial = '3'
file_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/'
fq = 4.2
roi_list = [0,1]
events = [1, 2, 3]  # list of eventulation
reanalyze = True

output_struct, F_corrected, DFF, cells_index, noise, time_indices, signal2noise, TA_all, df_bouts = load_output_struct(
    fishlabel, trial, file_path+'ML_pipeline_output/dataset/')

config_file, nTrials, frame_rate_2p = load_config_file(fishlabel, trial, index=0, path=file_path+'ML_pipeline_output/dataset/')
F, Fneu, spks, stat, ops, iscell = load_suite2p_outputs('190207_F1/3/', file_path+'suite2p_output/')

output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/Tau_calculation/'+fishlabel+'_'+trial+'/'
# os.mkdir(output_path)


roi = roi_list[1]
# cut and put in percentages
dff_cut = DFF[:,50:650]*100

output_path = output_path+str(roi)+'/'
os.mkdir(output_path)

try:
    df_kinematics = pd.read_pickle(output_path+'df_kinematics_roi'+str(roi))
except FileNotFoundError:
    df_kinematics = pd.DataFrame(columns=['Frame_peak', 'Frame_end_of_decay',
                                          'Tau', 'popt', 'pcov'],
                                 index=['event' + str(i) for i in events],
                                 dtype=object)

# Plot dff and event
plt.close()
plt.figure('DFF')
plt.plot(dff_cut[roi], 'b', label='DFF')
plt.legend()
plt.ginput(timeout=2)

if reanalyze:
    for event in events:
        event = str(event)
        output = get_frames_time_points(event, dff_cut[roi], fq, output_path)
        df_kinematics.loc['event'+event, 'Frame_peak'] = output[0]
        df_kinematics.loc['event'+event, 'Frame_end_of_decay'] = output[1]
        print('Summary event' + str(event))
        print(df_kinematics.loc['event'+event])


for event in events:
    plt.ion()
    event = str(event)
    tau, popt, pcov = tau_calculation(event, dff_cut[roi], df_kinematics, fq, output_path)
    save_tau = input('expo fit ok ? (yes)/no')  # ask user if the fit is ok
    if save_tau != 'no': # if it entered something else than no, save it
        df_kinematics.at['event'+event, 'Tau'] = tau
        df_kinematics.at['event' + event, 'popt'] = popt
        df_kinematics.at['event' + event, 'pcov'] = pcov
    else:  # if entered no, put np.nan value
        df_kinematics.at['event' + event, 'Tau'] = np.nan
        df_kinematics.at['event' + event, 'popt'] = np.nan
        df_kinematics.at['event' + event, 'pcov'] = np.nan

df_kinematics.to_pickle(output_path+'df_kinematics_roi'+str(roi))
df_kinematics.to_csv(output_path+'df_kinematics_roi'+str(roi)+'.csv')


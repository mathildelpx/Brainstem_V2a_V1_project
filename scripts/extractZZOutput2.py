import json
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle
from utils import functions_ZZ_extraction as fct
from utils.import_data import *
from utils import set_up_env as env


fishlabel = '200930_F1'
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'
raw_data_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ZZ_output/'
summary_file_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/summaryDataFinal.csv'


#   Read the summary csv file with info on all files stored.
try:
    summary_file = pd.read_csv(summary_file_path,
                               header=0)
except FileNotFoundError:
    print('The path to the summary file is not valid.')
    quit()

summary_fish = summary_file[summary_file['Fishlabel'] == fishlabel]
nFiles = len(summary_fish)

# load parameters of experiment
try:
    experiment = load_experiment(output_path, fishlabel)
    experiment = experiment[0]
except FileNotFoundError:
    print('Experiment for this fish was not found. Please create Exp object to reference experiment parameters.')
    quit()
fps_beh = experiment.fps_beh

# Create folders to store analysis
fct.create_analysis_env(output_path, fishlabel)
analysis_log = fct.create_analysis_log()

pd.options.mode.chained_assignment = None


# Load txt files from a folder
for i in range(nFiles):
    # Necessity given the struct of the output
    numWell = 0
    numBout = 0

    depth = summary_fish['Depth'].iloc[i]
    trial = str(int(summary_fish['Trial'].iloc[i]))
    print("depth", depth)
    print("trial ", trial)
    trial_path = raw_data_path + fishlabel + '/' + fishlabel + '_' + depth + '-' + trial

    #  Set-up environment to save output data for this plane
    env.set_up_env_plane(output_path, fishlabel, depth)

    #  Load the ZZ output txt file
    # If no txt file, go on to the next loop indent
    filepath = fct.load_zz_results(trial_path)
    if not filepath:
        continue

    # Open txt file as a struct
    with open(filepath) as f:
        big_supstruct = json.load(f)
        supstruct = big_supstruct["wellPoissMouv"][numWell][0]

    # Creating a DataFrame contaning information on bout, fish position... at each frame

    # Defining index of the DataFrame as the frame number
    # number of bouts in file
    NBout = len(supstruct)
    analysis_log['n_bouts'] = NBout
    print('Number of bouts:', NBout)

    # The last frames of the last bout
    # /!\ With Python you start indexing at 0, so the last bout is indexed as NBout -1
    try:
        End_Index = supstruct[NBout - 1]["BoutEnd"]
    except IndexError:
        print('No bout detected for this trial.')
        End_Index = int(input('Number of frames to build empty DataFrames ?'))

    # create index of dataframe: step is one frame
    # range(x) gives you x values, from 0 to x-1. So here you have to add +1
    index_frame = pd.Series(range(End_Index + 1))
    # Creating empty DataFrame
    df_frame = pd.DataFrame({'Name': fishlabel + '_' + depth + '-' + trial,
                             'Time_index': np.nan,
                             'BoutNumber': np.nan,
                             'Tail_angle': np.nan,
                             'Bend_Index': np.nan,
                             'Instant_TBF': np.nan,
                             'Bend_Amplitude': np.nan}, index=index_frame)

    # Filling this DataFrame

    # Creating a DataFrame containing start frame index and end frame index

    # Filling first the frames with a bout
    # using functions to find bout number and number of oscillations of this bout correspond to the frame
    # boutstart summed corresponds to the start frame of a given bout if all the videos were just one big video
    df_frame.Time_index = pd.Series(df_frame.index).apply(fct.get_time, args=(fps_beh,))
    df_frame.BoutNumber = pd.Series(df_frame.index).apply(fct.bout_num, args=(supstruct, NBout))
    df_frame.Tail_angle = pd.Series(df_frame.index).apply(fct.tail_angle, args=(supstruct, NBout))
    df_frame.Bend_Index = pd.Series(df_frame.index).apply(fct.bend_index, args=(supstruct, NBout))
    df_frame.Bend_Amplitude = pd.Series(df_frame.index).apply(fct.bend_amplitude, args=(supstruct, NBout))

    # Filling the frames without bouts
    # need to change this to the last value
    # df_frame['Tail_angle'].fillna(df_frame.Tail_angle.median(), inplace=True)
    # print('resting state tail angle: ', df_frame.Tail_angle.median())
    df_frame['Tail_angle'].fillna(0, inplace=True)

    # Creating another DataFrame containing quite the same info but per bout
    # index is the number of the bouts
    df_bout_index = pd.Series(range(NBout))
    num_osc = df_bout_index.apply(fct.N_osc_b, args=(supstruct,))
    bout_duration = df_bout_index.apply(fct.bout_duration, args=(supstruct, fps_beh))
    bout_start = df_bout_index.apply(fct.get_bout_start, args=(supstruct,))
    bout_end = df_bout_index.apply(fct.get_bout_end, args=(supstruct,))
    max_bend_amp = df_bout_index.apply(fct.max_bend_amp, args=(supstruct,))
    min_bend_amp = df_bout_index.apply(fct.min_bend_amp, args=(supstruct,))
    first_bend_amp = df_bout_index.apply(fct.first_bend_amp, args=(supstruct,))
    second_bend_amp = df_bout_index.apply(fct.second_bend_amp, args=(supstruct,))
    ratio_first_second_bend = df_bout_index.apply(fct.ratio_bend, args=(supstruct,))
    mean_TBF = df_bout_index.apply(fct.mean_tbf, args=(supstruct, fps_beh))
    iTBF = df_bout_index.apply(fct.bout_iTBF, args=(supstruct, df_frame))
    median_iTBF = df_bout_index.apply(fct.median_iTBF, args=(iTBF,))
    mean_tail_angle = df_bout_index.apply(fct.mean_tail_angle, args=(df_frame, supstruct))
    tail_angle_sum = df_bout_index.apply(fct.tail_angle_sum, args=(df_frame, supstruct))
    integral_tail_angle = df_bout_index.apply(fct.integral_ta, args=(df_frame, supstruct))

    df_bout = pd.DataFrame({'Name': pd.Series(fishlabel + '_' + depth + '-' + trial, index=df_bout_index),
                            'Manual_type': pd.Series('Others', index=df_bout_index),
                            'Bout_Duration': bout_duration,
                            'BoutStartVideo': bout_start,
                            'BoutEndVideo': bout_end,
                            'BoutStart_summed': bout_start,
                            'BoutEnd_summed': bout_end,
                            'Number_Osc': num_osc,
                            'Max_Bend_Amp': max_bend_amp,
                            'abs_Max_Bend_Amp': abs(max_bend_amp),
                            'Min_Bend_Amp': min_bend_amp,
                            'First_Bend_Amp': first_bend_amp,
                            'Second_Bend_Amp': second_bend_amp,
                            'Ratio First Second Bend': ratio_first_second_bend,
                            'mean_TBF': mean_TBF,
                            'iTBF': iTBF,
                            'median_iTBF': median_iTBF,
                            'mean_tail_angle': mean_tail_angle,
                            'Tail_angle_sum': tail_angle_sum,
                            'Integral_TA': integral_tail_angle,
                            'Side_biais': pd.Series(np.nan, index=df_bout_index)
                            }, index=df_bout_index, dtype=object)

    df_frame.to_pickle(output_path + 'dataset/' + fishlabel + '/' +
                       fishlabel + '_raw_frame_dataset_' + depth + '-' + trial)
    df_bout.to_pickle(output_path + 'dataset/' + fishlabel + '/' +
                      fishlabel + '_raw_bout_dataset_' + depth + '-' + trial)
    print('DataFrames saved to pickle')

    print('file', depth + '-' + trial, 'done')
    print('#################################""')

    with open(output_path + 'logs/' + fishlabel + '_' + trial + '_analysis_log', 'wb') as fp:
        pickle.dump(analysis_log, fp)
        print('Analysis log was saved in', fp.name)

    # Merge
    # if first file of the fish, create the overall dataframes as copy of the actual dataframes
    if i == 0:
        df_bout_all = df_bout.copy()
        df_frame_all = df_frame.copy()
    # if not, you actualize the index of BoutStart and BoutEnd by adding the number of index frm previous DataFrames_all
    else:
        df_bout.BoutStart_summed += len(df_frame_all)
        df_bout.BoutEnd_summed += len(df_frame_all)
        df_bout_all = df_bout_all.append(df_bout, ignore_index=True, sort=None)
        df_frame_all = df_frame_all.append(df_frame, ignore_index=True, sort=None)


######## END OF LOOP

# Save the resume file for this fish with info on the experiments and trials.
print(analysis_log)
with open(output_path + 'logs/' + fishlabel + '_' + trial + '_analysis_log', 'wb') as fp:
    pickle.dump(analysis_log, fp)
    print('Analysis log was saved in', fp.name)

# Save overall DataFrames to pickle
df_frame_all.to_pickle(output_path + 'dataset/' + fishlabel + '/' + df_frame_all)
df_bout_all.to_pickle(output_path + 'dataset/' + fishlabel + '/' + df_bout_all)


#  Plot first bouts of this fish to have a quick look of what behavior looked like
plt.figure(1)
plt.suptitle('Tail angle over frames for each bout')
if NBout >= 25:
    # Plot the 25 first bouts of this fish. To have an idea of what the tracking and the behavior looks like.
    for i, index in enumerate(range(25)):
        plt.subplot(5, 5, i + 1)
        plt.plot(df_frame_all.Tail_angle[df_bout_all.BoutStart_summed[index]:df_bout_all.BoutEnd_summed[index]])
        plt.plot(df_frame_all.Bend_Amplitude[df_bout_all.BoutStart_summed[index]:df_bout_all.BoutEnd_summed[index]],
                 'rx', markersize=1.5)
        plt.ylim(-50, 50)
        plt.title(index)
        if i == 20:
            plt.xlabel('frame')
            plt.ylabel('Tail angle')
    plt.savefig(output_path + 'fig/' + fishlabel + '/Tail_angle_traces.pdf', transparent=True)
else:
    for i, index in enumerate(range(16)):
        plt.subplot(4, 4, i + 1)
        plt.plot(df_frame_all.Tail_angle[df_bout_all.BoutStart_summed[index]:df_bout_all.BoutEnd_summed[index]])
        plt.plot(df_frame_all.Bend_Amplitude[df_bout_all.BoutStart_summed[index]:df_bout_all.BoutEnd_summed[index]],
                 'rx', markersize=1.5)
        plt.ylim(-50, 50)
        plt.title(index)
        if i == 12:
            plt.xlabel('frame')
            plt.ylabel('Tail angle')
    plt.savefig(output_path + 'fig/' + fishlabel + '/Tail_angle_traces.pdf', transparent=True)


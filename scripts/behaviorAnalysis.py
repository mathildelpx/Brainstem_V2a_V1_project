import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import time
from utils.import_data import *
from utils.load_classified_data import *
from utils.functions_behavior import *
from utils.plotting import *


pd.options.mode.chained_assignment = None

fishlabel = '190515_F2'
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'
classification_path = '/home/mathilde.lapoix/Bureau/boutsClusteringMathilde/mathildeFinal/output_dataframe'

# LOAD CONFIG FILES, STRUCTS

trial = str(input('Trial num ?'))

try:
    trials_correspondence = load_trials_correspondence(output_path, fishlabel)
except (FileNotFoundError, EOFError):
    pass

analysis_log = load_analysis_log(output_path, fishlabel, trial)
experiment = load_experiment(output_path, fishlabel)
experiment = experiment[0]
analysis_log['date_analysis'] = time.strftime("%d/%m/%Y")

# define depth and filename of this plane:
if analysis_log['depth'] is np.nan:
    index = find_indices(trials_correspondence['Trial_num'], lambda e: e == trial)[0]
    depth = trials_correspondence['Depth'][index]
    analysis_log['depth'] = depth
else:
    depth = analysis_log['depth']

if analysis_log['filename'] is np.nan:
    index = find_indices(trials_correspondence['Trial_num'], lambda e: e == trial)[0]
    filename = trials_correspondence['Filename'][index]
    analysis_log['filename'] = filename
else:
    filename = analysis_log['filename']

fps_beh = experiment.fps_beh
fps_2p = experiment.fps_2p
# import pickles files for one fish
try:
    raw_frame_dataset = pd.read_pickle(output_path + 'dataset/' + fishlabel + '/' + trial + '/raw_frame_dataset_' + str(trial))
    raw_bout_dataset = pd.read_pickle(output_path + 'dataset/' + fishlabel + '/' + trial + '/raw_bout_dataset_' + str(trial))
except FileNotFoundError:
    raw_frame_dataset = pd.read_pickle(output_path + 'dataset/' + fishlabel + '/' + fishlabel + '_raw_frame_dataset_' + trial)
    raw_bout_dataset = pd.read_pickle(output_path + 'dataset/' + fishlabel + '/' + fishlabel + '_raw_bout_dataset_' + trial)
nBouts=len(raw_bout_dataset)
raw_bout_dataset['Keep'] = 1
analyzed_bout_dataset = raw_bout_dataset.copy()

print("trial ", trial, "\n depth", depth)
print('frame rate behavior:', fps_beh)
print('frame rate 2P:', fps_2p)
print('Total number of bouts: ', nBouts)

##################################################"

# LOAD CLASSIFICATION
# filename = trials_correspondence['Filename'][index]+'-2'
# NB: check that filename is actually corresponding (is doublon you need to add '-2'
classification_df = load_output_dataframe(classification_path, filename)
if nBouts != len(classification_df):
    print('number of bouts not corresponding between the data already analyzed by the pipeline and the classified data.')
    print('check file name, doublons or date of analysis/acquisition.')
analyzed_bout_dataset['classification'] = get_classification(list(analyzed_bout_dataset.index), classification_df)
analyzed_bout_dataset['category'] = pd.Series(analyzed_bout_dataset.index).apply(replace_category,
                                                                      args=(analyzed_bout_dataset, raw_frame_dataset, 10))


# should remove this step in further analysis
try:
    analyzed_bout_dataset['abs_Max_Bend_Amp']
except KeyError:
    analyzed_bout_dataset['abs_Max_Bend_Amp'] = abs(analyzed_bout_dataset.Max_Bend_Amp)

for i in range(1, 12):
    if i * i >= len(raw_bout_dataset):
        output = i
        break
    elif len(raw_bout_dataset) == 1:
        continue
n_col, n_row = (output, output)

# plot bouts with class to check it
colors = ['#FF00FF', '#FF8000', '#FF8C00', '#228B22']
# get max bout duration to know how long the fixed time scale needs to be
max_bout_duration = analyzed_bout_dataset['Bout_Duration'].max()
plt.figure(35, figsize=[25,15])
plt.suptitle('Tail angle over frames for each bout')
for i, bout in enumerate(range(nBouts)):
    plt.subplot(n_row, n_col, i + 1)
    classification = analyzed_bout_dataset.classification[bout]
    if classification == 0:
        cat = analyzed_bout_dataset.category[bout]
        # turns on the right side will be first color, here pink
        # if left turns, orange (second color)
        if cat == 'R':
            color = colors[0]
        else:
            color = colors[1]
    else:
        try:
            color = colors[int(classification)+2]
        # if bout was not classified (ValueError) or flagged (IndexError)
        except (ValueError, IndexError):
            print('bout:', bout)
            print('class:', classification)
            color = 'black'
    # plot each bout from start, with a fixed time scale (take the max bout duration to scale every other).
    start = analyzed_bout_dataset.BoutStartVideo[bout]
    end = start+int(max_bout_duration*fps_beh)
    plt.plot(raw_frame_dataset.Time_index[start:end], raw_frame_dataset.Tail_angle[start:end], color=color)
    # Because bend index was calculated using matlab, the indexing is shifted from one (matlab starts indexing at 1,
    # Python at 0. Therefore you need to shift back the bend index.
    # When it comes to time, you shift from one time step: 1/fq_beh seconds.
    plt.plot(raw_frame_dataset.Time_index[start:end]-(1/fps_beh), raw_frame_dataset.Bend_Amplitude[start:end], 'rx', markersize=1.5)
    plt.ylim(-70,70)
    plt.title('B '+ str(i))
    if i == 0:
        plt.ylabel('Tail angle [°]')
    if i == ((n_row-1)*n_col):
        plt.xlabel('Time (s)')
        plt.ylabel('Tail angle')
plt.tight_layout()
plt.savefig(output_path + 'fig/' + fishlabel + '/' + trial + '/Traces_all_bouts_' + trial + '.png', transparent=True)
plt.savefig(output_path + 'fig/' + fishlabel + '/' + trial + '/Traces_all_bouts_' + trial + '.pdf', transparent=True)
plt.show()

# Plot, for each categories, all bouts on top of each other
for category in ['R', 'L', 'F', 'O', np.nan]:
    print(category)
    plot_categories(category, analyzed_bout_dataset, raw_frame_dataset, fps_beh,
                    output_path + 'fig/' + fishlabel + '/' + trial + '/')
    print('done')

# Exclude bouts if needed

exclusion = [int(x) for x in input('Bouts to exclude?').split()]
if exclusion:
    analyzed_bout_dataset.loc[exclusion, 'Keep'] = 0
    analyzed_bout_dataset = analyzed_bout_dataset.drop[np.where(analyzed_bout_dataset.Keep == 0)[0]]

analysis_log['bouts_excluded'] = exclusion


# Figure to show kinematics parameters per category

df_bouts = analyzed_bout_dataset.copy()
# replace missing values in classification and category to see also the bouts with missing classification
df_bouts['classification'] = df_bouts['classification'].replace(np.nan, 3.0)
plot_violin_kinematics_class(df_bouts, 2, 3, output_path, fishlabel, trial)
df_bouts['category'] = df_bouts['category'].replace(np.nan, 'None')
plot_violin_kinematics_cat(df_bouts, 2, 3, output_path, fishlabel, trial)


# Build behavior trace for suite2p
ta_trace = str(input('Do you want to save TA array for suite2P visualisation? y/()'))
if ta_trace:
    TA = np.zeros((len(raw_frame_dataset), 2))
    TA[:,0] = raw_frame_dataset.Time_index
    TA[:,1] = raw_frame_dataset.Tail_angle
    old_fq = str(input('Type in time series format the frequency of behavior acquisition: (example: 0.005S'))
    new_fq = str(input('Type in time series format the new frequency you want to rescale the behavior to:'))
    nFrames = int(input('Number of frames in calcium imaging ?'))
    behavior_trace = behavior_resampled(TA, old_fq, new_fq, nFrames, output_path + 'np_array/' + fishlabel + '/' + trial + '/')

# Create a list of objects from class Bout and save it
create_bout_objects(df_bouts, raw_frame_dataset, output_path + 'dataset/' + fishlabel + '/' + trial + '/')

# SAVE
raw_frame_dataset.to_pickle(output_path + 'dataset/' + fishlabel + '/' + trial + '/analyzed_frame_dataset')
analyzed_bout_dataset.to_pickle(output_path + 'dataset/' + fishlabel + '/' + trial + '/analyzed_bout_dataset')
print('Dataframes saved in pickle format.')

with open(output_path + 'logs/' + fishlabel + '_' + trial + '_analysis_log', 'wb') as fp:
    pickle.dump(analysis_log, fp)
    print('Analysis log was saved in', fp.name)
with open(output_path + 'dataset/' + fishlabel + '/resume_fish_' + fishlabel, 'wb') as handle:
    pickle.dump(trials_correspondence, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Trials correspondence saved in pickle')


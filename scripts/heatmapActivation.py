import numpy as np
import pandas as pd
import pickle
from utils.functions_calcium_analysis import *
from utils.plotting import *
from utils.heatmap_activation import *
from tools.list_tools import *

####################################################################################################

fishlabel = '200813_F1'
depth = '160um'
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'
data_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/suite2p_output/'
bg_path = '/home/mathilde.lapoix/Documents/RSN_project/Data/backgrounds/' + fishlabel + '/' + depth + '_background.png'
reanalyze = False

############################################################################

with open(output_path + 'dataset/' + fishlabel + '/' + depth + '/struct', 'rb') as output:
    output_struct = pickle.load(output)
analysis_log = load_analysis_log(output_path, fishlabel, depth)
experiment = load_experiment(output_path, fishlabel)[0]

with open(output_path + 'dataset/' + fishlabel + '/df_frame_plane_' + depth, 'rb') as f:
    df_frame = pickle.load(f)

fps_2p = experiment.fps_2p
fps_beh = experiment.fps_beh
mean_time_delta_2p = 1 / fps_2p

window_max_DFF = analysis_log['window_max_dff']
F_corrected, DFF = output_struct['F_corrected'], output_struct['DFF']
cells_index, noise = output_struct['cells_index'], output_struct['noise']
time_indices = np.array(output_struct['time_indices'])
signal2noise = output_struct['signal_noise']
TA_all, df_bouts = output_struct['TA_all'], output_struct['df_bouts']
F = output_struct['F']
Fneu = output_struct['Fneu']
spks = output_struct['spks']
stat = output_struct['stat']
ops = output_struct['ops']
iscell = output_struct['iscell']
filtered_dff = output_struct['filtered_dff']
filtered_dff = np.array(filtered_dff)
filtered_noise = output_struct['filtered_noise']

nROI, nFrames = F.shape

# create time points to plot everything according to time and not frame index.
time_indices = [frame * (1 / fps_2p) for frame in range(nFrames)]

output_struct['fps_beh'] = fps_beh
output_struct['fps_2p'] = fps_2p
output_struct['output_path'] = output_path
output_struct['bg_path'] = bg_path
output_struct['fishlabel'] = fishlabel
output_struct['depth'] = depth
output_struct['time_indices'] = time_indices
output_struct['df_frame'] = df_frame
output_struct['colors_cat'] = {'F': '#FFCBDD',
                               'R': '#FB4B4E',
                               'L': '#7C0B2B',
                               'S': '#3E000C',
                               'Exc': '#BE6B84'}

for bout in df_bouts.index:
    plot_heatmap_bout(bout, output_struct, 1, 'raw_DFF')
    plot_heatmap_bout(bout, output_struct, 1, 'filtered_DFF')
    heatmap_recruitment(bout, output_struct, 1, 'filtered_dff')
    # plot_heatmap_noise_bout(bout, ops, stat, filtered_dff, filtered_noise, cells_index, bg_path, 1, 'filtered_DFF')
    # plot_heatmap_noise_bout(bout, ops, stat, DFF, noise, cells_index, bg_path, 1, 'raw_DFF')
    # plot_heatmap_integral_bout(bout, ops, stat, DFF, cells_index, bg_path, 1, 'raw_DFF')
    # plot_heatmap_integral_bout(bout, ops, stat, filtered_dff, cells_index, bg_path, 1, 'filtered_DFF')

for category in get_unique_elements(df_bouts.category):
    plot_heatmap_mean_activity_cat(category, output_struct, 1, 'filtered_DFF')
    plot_heatmap_median_activity_cat(category, output_struct, 1, 'filtered_DFF')
    plot_heatmap_std_activity_cat(category, output_struct, 1, 'filtered_DFF')


with open(output_path + 'logs/' + fishlabel + '_' + depth + '_analysis_log', 'wb') as fp:
    pickle.dump(analysis_log, fp)
    print('Analysis log was saved in', fp.name)
with open(output_path + 'dataset/' + fishlabel + '/' + depth + '/struct', 'wb') as output:
    pickle.dump(output_struct, output, protocol=pickle.HIGHEST_PROTOCOL)
    print('Output struct was saved in', output.name)

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt


# function to return a list of index corresponding to element in a list (lst) filling a condition
def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def load_output(fishlabel, trial, output_path):
    try:
        with open(output_path + 'dataset/' + fishlabel + '/config_file', 'rb') as f:
            config_file = pickle.load(f)
        with open(output_path + 'dataset/' + fishlabel + '/' + trial + '/struct', 'rb') as s:
            output_struct = pickle.load(s)
    except FileNotFoundError:
        print('Missing analysis')
        quit()
    return config_file, output_struct


fishlabel = '190306_F1'
trial = '8'
index= 7
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'

config_file, output_struct = load_output(fishlabel, trial, output_path)

nTrials, frame_rate_2p, fq = len(config_file['Trial_num']), config_file['Frame_rate_2p'][index], config_file['Fq_camera'][index]
mean_time_delta_2p = config_file['Time_step_2p'][index]
threshold_bout, window_max_DFF = config_file['Time_threshold_bouts'][index], config_file['Time_window_max_DFF'][index]
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
F0_inf = config_file['F0_inf'][index]
F0_sup = config_file['F0_sup'][index]
# noise = np.empty_like(iscell)
# for ROI in cells_index:
#     print(ROI)
#     print(np.std(DFF[ROI, F0_inf:F0_sup]))
#     noise[ROI,0] = np.std(DFF[ROI, F0_inf:F0_sup])
#     print(noise[ROI,0])

# correct the DFF to only have the values for the ROI that are cells
DFF[iscell[:,0] == 0] = np.nan
# create a transposed DFF to plot it easily with matplotlib.
DFF_T = np.transpose(DFF)

filtered_dff = output_struct['filtered_dff']
filtered_noise = output_struct['filtered_noise']
chosen_window = output_struct['chosen_window']

arr_dff = np.array(filtered_dff)

fig_signal2noise_3 = pickle.load(open(os.getcwd() + '/analyzed_data/fig/' + fishlabel +
                                      '/' + trial + '/pickle_signal2noise_runavg3',
                                      'rb'))

fig_signal2noise_5 = pickle.load(open(os.getcwd() + '/analyzed_data/fig/' + fishlabel +
                                      '/' + trial + '/pickle_signal2noise_runavg5',
                                      'rb'))

fig_dispersion = pickle.load(open(os.getcwd() + '/analyzed_data/fig/' + fishlabel +
                                  '/' + trial + '/pickle_maxDFF_dispersion_runavg'+str(chosen_window),
                                  'rb'))

fig_digit = pickle.load(open(os.getcwd() + '/analyzed_data/fig/' + fishlabel +
                             '/' + trial + '/pickle_digit',
                             'rb'))

fig_min_max_dot = pickle.load(open(os.getcwd() + '/analyzed_data/fig/' + fishlabel +
                               '/' + trial + '/pickle_maxDFF_min_max_dot_sized',
                               'rb'))

fig_min_max = pickle.load(open(os.getcwd() + '/analyzed_data/fig/' + fishlabel +
                               '/' + trial + '/pickle_maxDFF_min_max',
                               'rb'))

plt.show()





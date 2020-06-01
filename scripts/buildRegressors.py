import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_regressor(tau_value, df_bouts, TA_all):
    regressor = np.zeros(TA_all[:, 0].shape)
    regressor_left = regressor.copy()
    regressor_right = regressor.copy()

    def exp_function(x):
        return np.exp(x / -tau_value)

    for bout in df_bouts.index:
        start = df_bouts.BoutStartVideo[bout]
        try:
            end_decay = df_bouts.BoutStartVideo[bout + 1]
        except KeyError:
            end_decay = len(regressor)
        x_range = pd.Series([x / fq for x in range(0, end_decay - start)])
        shape = x_range.apply(exp_function, args=())
        regressor[start:end_decay] = shape
        if df_bouts.Max_Bend_Amp[bout] < 0:
            regressor_right[start:end_decay] = shape
        else:
            regressor_left[start: end_decay] = shape
    return regressor, regressor_left, regressor_right


output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'
fishlabel = '190104_F2'
trial = '7'
index= 11

try:
    with open(output_path + 'dataset/' + fishlabel + '/config_file', 'rb') as f:
        config_file = pickle.load(f)
    with open(output_path + 'dataset/' + fishlabel + '/' + trial + '/struct', 'rb') as s:
        output_struct = pickle.load(s)
except FileNotFoundError:
    print('Missing analysis')
    quit()

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
shift = output_struct['shift']
bad_frames = output_struct['bad_frames']
F0_inf = config_file['F0_inf'][index]
F0_sup = config_file['F0_sup'][index]
filtered_dff = output_struct['filtered_dff']
filtered_noise = output_struct['filtered_noise']
chosen_window = output_struct['chosen_window']

# correct the DFF to only have the values for the ROI that are cells
DFF[iscell[:,0] == 0] = np.nan
# create a transposed DFF to plot it easily with matplotlib.
DFF_T = np.transpose(DFF)
arr_dff = np.array(filtered_dff)

regressor, regressor_left, regressor_right = build_regressor(0.2, df_bouts, TA_all)

# make clusters of cells
cells_left_cluster = [0, 1, 5, 6, 9, 20]
cells_right_cluster = [2, 4, 7, 11, 13, 34, 221, 223]
subcluster1 = [0, 1, 20]
subcluster2 = [11, 2, 4, 221]

# for trial 6
# inf = 25833
# sup = 29761
inf = 10000
sup = 18000

plt.figure()
plt.plot(TA_all[inf:sup, 0], TA_all[inf:sup, 1], color='#a37871')
plt.xlabel('Time [s]')
plt.ylabel('Tail angle [°]')
plt.close()

plt.figure()
plt.plot([x/fq for x in range(inf,sup)], regressor[inf:sup], color='green', label='all bouts')
plt.plot([x/fq for x in range(inf,sup)], regressor_left[inf:sup]-1.5, color='cyan', label='left')
plt.plot([x/fq for x in range(inf,sup)], regressor_right[inf:sup]-3, color='magenta', label='right')
plt.xlabel('Time [s]')
plt.ylabel('DF/F')
plt.legend()

# for 190104_F2_6
# cells_left_cluster = [20,753,194,14,198]
# cells_right_cluster = [11,546,8,31,7,9]
# for same fish trial 7
cells_left_cluster = [12, 16, 17, 36]
cells_right_cluster = [11, 30, 32, 161]

inf_2p = int((inf/fq)*frame_rate_2p)
sup_2p = int((sup/fq)*frame_rate_2p)

plt.figure('Left cluster')
for i, cell in enumerate(cells_left_cluster):
    plt.plot([x/frame_rate_2p for x in range(inf_2p, sup_2p)], arr_dff[cell, inf_2p:sup_2p]+5*i, label='cell '+str(cell), color='cyan')
plt.xlabel('Time [s]')
plt.ylabel('DF/F')
plt.legend()
plt.figure('Right cluster')
for i, cell in enumerate(cells_right_cluster):
    plt.plot([x/frame_rate_2p for x in range(inf_2p, sup_2p)], arr_dff[cell, inf_2p:sup_2p]+5*i, label='cell '+str(cell), color='magenta')
plt.xlabel('Time [s]')
plt.ylabel('DF/F')
plt.legend()

# for each cell, plot its trace upon the regressors
for cell in cells_left_cluster:
    plt.figure(str(cell))
    plt.title('In left cluster: cell'+ str(cell))
    plt.plot([x / fq for x in range(inf, sup)], regressor[inf:sup], color='green', label='all bouts')
    plt.plot([x / fq for x in range(inf, sup)], regressor_left[inf:sup] - 2, color='cyan', label='left')
    plt.plot([x / fq for x in range(inf, sup)], regressor_right[inf:sup] - 4, color='magenta', label='right')
    plt.plot([x/frame_rate_2p for x in range(inf_2p, sup_2p)], arr_dff[cell, inf_2p:sup_2p], label='DFF', color='k')
    plt.plot([x/frame_rate_2p for x in range(inf_2p, sup_2p)], arr_dff[cell, inf_2p:sup_2p]-2, color='k')
    plt.plot([x/frame_rate_2p for x in range(inf_2p, sup_2p)], arr_dff[cell, inf_2p:sup_2p]-4, color='k')
    plt.legend()
    plt.savefig(output_path+'fig/'+fishlabel+'/'+trial+'/dff_against_reg_cell_'+str(cell)+'.png')

for cell in cells_right_cluster:
    plt.figure(str(cell))
    plt.title('In right cluster: cell'+ str(cell))
    plt.plot([x / fq for x in range(inf, sup)], regressor[inf:sup], color='green', label='all bouts')
    plt.plot([x / fq for x in range(inf, sup)], regressor_left[inf:sup] - 2, color='cyan', label='left')
    plt.plot([x / fq for x in range(inf, sup)], regressor_right[inf:sup] - 4, color='magenta', label='right')
    plt.plot([x/frame_rate_2p for x in range(inf_2p, sup_2p)], arr_dff[cell, inf_2p:sup_2p], label='DFF', color='k')
    plt.plot([x/frame_rate_2p for x in range(inf_2p, sup_2p)], arr_dff[cell, inf_2p:sup_2p]-2, color='k')
    plt.plot([x/frame_rate_2p for x in range(inf_2p, sup_2p)], arr_dff[cell, inf_2p:sup_2p]-4, color='k')
    plt.legend()
    plt.savefig(output_path+'fig/'+fishlabel+'/'+trial+'/dff_against_reg_cell_'+str(cell)+'.png')


### FROM STYTRA

def calcium_kernel(tau):
    return lambda x: np.exp(-x/(tau/np.log(2)))


def convolve_regressors(regressor, kernel):
    return np.convolve(regressor, kernel)[0:len(regressor)]


ker = calcium_kernel(0.2)(np.arange(0,3,1/frame_rate_2p))

# resample

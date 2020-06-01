import os
import time
import numpy as np
import pandas as pd
import json
from utils.import_data import *
from utils.functions_calcium_analysis import *
from utils.plotting import *


# function to return a list of index corresponding to element in a list (lst) filling a condition
def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


class ReAnalyze(Exception):
    pass


class NoBehavior(Exception):
    pass


####################################################################################################

fishlabel = '190104_F2'
trial = '8'
shift = 1
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'
data_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/suite2p_output/'
reanalyze = False

analysis_log = load_analysis_log(output_path, fishlabel, trial)
experiment = load_experiment(output_path, fishlabel)[0]
# experiment = experiment[0]

for foldername in ['dataset', 'fig', 'np_array', 'csv_files']:
    try:
        os.mkdir(output_path + foldername + '/' + fishlabel + '/')
    except FileExistsError:
        pass
    try:
        os.mkdir(output_path + foldername + '/' + fishlabel + '/' + trial + '/')
    except FileExistsError:
        pass

# window of time to look at calcium signal around the bout
# window of time which must separate two bouts, otherwise they are too close and are excluded from the analysis
fps_2p = experiment.fps_2p
fps_beh = experiment.fps_beh
analysis_log['date_analysis'] = time.strftime("%d/%m/%Y")

############################################################################
# Load outputs from suite2P
F, Fneu, spks, stat, ops, iscell = load_suite2p_outputs(fishlabel, trial, data_path)
# load output struct if exisiting
output_struct = load_output_struct(fishlabel, trial, output_path+'dataset/', data_path)

nFrames, nROIs = output_struct['nFrames'], output_struct['nROI']
# create time points to plot everything according to time and not frame index.
time_indices = [frame*(1/fps_2p) for frame in range(nFrames)]


##### CORRECT FLUORESCENCE

# Define automatic detection of motion artifact: on what parameter do you rely ?
condition_movement = ops['xoff'] > np.percentile(ops['xoff'], 3)
bad_frames = np.where(condition_movement)

F_corrected, cells_index, del_frames = correct_2p_outputs(fishlabel, trial, output_path, bad_frames, reanalyze,
                                                          analysis_log)

analysis_log['removed_frames'] = del_frames
analysis_log['condition_ma'] = "ops['xoff'] > np.percentile(ops['xoff'], 3)"


###### DFF CALCULATION

# Load behavior
df_bouts, TA_all = load_behavior(fishlabel, trial, output_path)

# Layout for the figure of fluorescence
layout_F = (go.Layout(title=go.layout.Title(text='Corrected fluo over time', x=0),
                      yaxis=go.layout.YAxis(title='Fluoresence'),
                      xaxis= go.layout.XAxis(title='Time')))


# Try to load DFF and threshold list if already analyzed.
try:
    DFF = np.load(output_path + 'np_array/' + fishlabel + '/' + trial + '/DFF_' + trial + '.npy')
    noise = np.load(output_path + 'np_array/' + fishlabel + '/' + trial + '/noise_' + trial + '.npy')
    print('Loading existing DFF and noise')
    if reanalyze: raise ReAnalyze
# If not existing, calculate it.
except (FileNotFoundError, ReAnalyze):
    print('Calculating DFF and noise')
    print('Plotting corrected fluorescence...')
    plot_fluo(F_corrected, TA_all, cells_index, time_indices, layout_F, trial, output_path)

    # if analysis already contains defined limits, asks user if wants to reuse it or define new ones.
    if analysis_log['baseline_lim'] != (np.nan, np.nan):
        print('Exisiting limits for baseline calculation are:')
        print(analysis_log['baseline_lim'])
        reuse = str(input('Reuse them ? [y]/n'))
    else:
        reuse = 'n'

    # if user wants to redefine baseline limits, asks which methods to use (click on graph or type in)
    if reuse == 'n':
        method = int(input('Method to calculate DFF: click (0) or manual input (1)'))
        if method == 0:
            DFF, analysis_log = calc_DFF_click(F_corrected, cells_index, fps_2p, analysis_log)
        elif method == 1:
            DFF, analysis_log = calc_DFF_manual(F_corrected, cells_index, analysis_log)
    else:
        DFF = calc_DFF_reuse(F_corrected, cells_index, analysis_log)

    # calculate noise
    noise = calc_noise(DFF, cells_index, analysis_log)

    # save
    np.save(output_path + 'np_array/' + fishlabel + '/' + trial + '/DFF_' + trial + '.npy', DFF)
    np.save(output_path + 'np_array/' + fishlabel + '/' + trial + '/noise_' + trial + '.npy',
            noise)


# Create layout for DFF plotting
layout_DFF = (go.Layout(title =go.layout.Title(text='Delta fluo over time', x=0),
                        yaxis=go.layout.YAxis(title='DFF', showgrid=False, zeroline=False),
                        xaxis= go.layout.XAxis(title='Time', showgrid=False, zeroline=False)))

# plot dff
plot_dff(DFF, TA_all, time_indices, cells_index, shift, trial, layout_DFF, output_path)

# DFF is an array of shape (number of cells, number of frames)
# transpose the DFF array so the format is compatible with matplotlib library which plots columns as a function of rows
# DFF_T will be an array of shape (number of frames, number of cells)
# plt.plot function will plot columns one by one (DFF signal cell by cell)
DFF_T = np.transpose(DFF)
plt.figure('Raw')
plt.plot(time_indices, DFF_T)
plt.title('raw DFF of all ROIs')

# Load everything in a big structure
output_struct['shift'] = shift
output_struct['F_corrected'] = F_corrected
output_struct['DFF'] = DFF
output_struct['cells_index'] = cells_index
output_struct['noise'] = noise
output_struct['time_indices'] = time_indices
output_struct['df_bouts'], output_struct['TA_all'] = df_bouts, TA_all
output_struct['nROI'], output_struct['nFrames'] = nROIs, nFrames
output_struct['F0_inf'], output_struct['F0_sup'] = analysis_log['baseline_lim']
# calculate the ratio signal/noise for each cell
# it is a marker of how much you can rely on this cell signal
signal2noise = np.zeros((len(iscell),1))
signal2noise[:] = np.nan
for cell in cells_index:
    signal2noise[cell] = np.nanmean(DFF[cell]/noise[cell])
output_struct['signal_noise'] = signal2noise

# Save analysis log and output structure
with open(output_path + 'logs/' + fishlabel + '_' + trial + '_analysis_log', 'wb') as fp:
    pickle.dump(analysis_log, fp)
    print('Analysis log was saved in', fp.name)
with open(output_path + 'dataset/' + fishlabel + '/' + trial + '/struct', 'wb') as output:
    pickle.dump(output_struct, output, protocol=pickle.HIGHEST_PROTOCOL)
    print('Output struct was saved in', output.name)

################################################################"

###### Filter DF/F

time_indices = np.array(time_indices)
bouts = load_bout_object(output_path, fishlabel, trial)


def filtering_analysis(windows, cells_interest):
    for i, window in enumerate(windows):
        filtered_dff, filtered_noise = run_avg_filter(DFF, noise, window, cells_index, analysis_log)

        fig, (ax1, ax2) = plt.subplots(1, 2, num='Raw vs run avg ' + str(window), sharex=True, sharey=True,
                                       figsize=(10, 5))
        ax1.plot(time_indices, DFF_T[:, cells_index])
        ax1.set_title('Raw DF/F')
        ax1.set_xlabel('Time [sec]')
        ax1.set_ylabel('Raw DF/F')
        ax2.plot(time_indices, np.transpose(filtered_dff.iloc[cells_index]))
        ax2.set_title('Run avg ' + str(window))
        ax2.set_ylabel('Filtered DF/F')
        plt.savefig(output_path + 'fig/' + fishlabel +
                    '/' + trial + '/rawVSfilter_' + 'run_avg' + '_' + str(window),
                    format='png')
        with open(output_path + 'fig/' + fishlabel +
                  '/' + trial + '/pickle_rawVSfilter_' + 'run_avg' + '_' + str(window), 'wb') as fp:
            pickle.dump(fig, fp, protocol=pickle.HIGHEST_PROTOCOL)

        fig3 = plt.figure('All windows per cells')
        for j, cell in enumerate(list(cells_interest)):
            plt.subplot(2, 2, j + 1)
            if i == 0:
                plt.plot(time_indices, DFF[cell], color='orange', label='Raw')
            plt.plot(time_indices, filtered_dff.iloc[cell], label='window=' + str(window))
            plt.plot([0, np.max(time_indices)], [filtered_noise[cell] * 2, filtered_noise[cell] * 2], label='threshold w=' + str(window))
            plt.title('Cell ' + str(cell))
            plt.ylim(dff_min, dff_max)
        plt.subplots_adjust(hspace=0.35)
        plt.legend(bbox_to_anchor=(1.1, 1.05), ncol=1)
        plt.savefig(output_path + 'fig/' + fishlabel +
                    '/' + trial + '/filtered_dff_run_avg_test_cells',
                    format='png')
        with open(output_path + 'fig/' + fishlabel +
                    '/' + trial + '/pickle_runavg_test', 'wb') as fp:
            pickle.dump(fig3, fp, protocol=pickle.HIGHEST_PROTOCOL)
    plt.show()
    return cells_interest


def final_filtering(chosen_window):
    if chosen_window >= 5:
        list_w = [chosen_window-2, chosen_window, chosen_window+2]
    elif chosen_window == 3:
        list_w = [chosen_window, chosen_window+2]
    else:
        print('Not usable number of frames for running average. Please select odd value higher or equals to 3.')
        quit()
    for window in list_w:
        print(window)
        filtered_dff, filtered_noise = run_avg_filter(DFF, noise, window, cells_index, analysis_log)
        filtered_dff = np.array(filtered_dff)
        fig1, (ax11, ax12) = plt.subplots(1,2, num='Signal2Noise '+str(window), figsize=(10,5))
        plt.title('Max amplitude of cells during bouts as a function of their noise')
        ax11.set_title('Raw signal')
        ax11.set_xlabel('Noise')
        ax11.set_ylabel('max DF/F')
        # plot for each bout, the max DF/F as a function of noise
        for cell in cells_index:
            ax11.scatter([noise[cell]] * len(bouts),
                        pd.Series(bouts).apply(max_DFF,
                                                        args=(cell, filtered_dff, time_indices, experiment, analysis_log)))
        # get the max of the graph in xaxis to plot a line on all trace
        inf_raw = np.min(noise[cells_index])
        sup_raw = np.max(noise[cells_index])
        noise_b_raw = [inf_raw, sup_raw]
        ax11.plot(noise_b_raw, [e * 2 for e in noise_b_raw], label='2*noise')

        ax12.set_title('Filtered signal run avg ' + str(window))
        for cell in cells_index:
            ax12.scatter([filtered_noise[cell]]*len(bouts),
                        pd.Series(bouts).apply(max_DFF,
                                                        args=(cell, filtered_dff, time_indices, experiment, analysis_log)))
        print('filtered noise', filtered_noise[cell])
        print('max', max_DFF(bouts[0], cell, filtered_dff, time_indices, experiment, analysis_log))
        inf_filter = np.min(filtered_noise[cells_index])
        sup_filter = np.max(filtered_noise[cells_index])
        noise_b_filter = [inf_filter, sup_filter]
        ax12.plot(noise_b_filter, [e * 2 for e in noise_b_filter], label='2*noise')
        ax12.legend()
        fig1.savefig(output_path + 'fig/' + fishlabel +
                     '/' + trial + '/signal2noise_runavg_' + str(window),
                     format='png')
        with open(output_path + 'fig/' + fishlabel +
                  '/' + trial + '/pickle_signal2noise_runavg'+str(window), 'wb') as fp:
            pickle.dump(fig1, fp, protocol=pickle.HIGHEST_PROTOCOL)
        plt.show()
    filtered_dff, filtered_noise = run_avg_filter(DFF, noise, chosen_window, cells_index, analysis_log)
    return filtered_dff, filtered_noise


def dispersion_signal(cell, filtered_dff):
    arr_dff = np.array(filtered_dff)
    std_max = np.std(pd.Series(bouts).apply(max_DFF, args=(cell, arr_dff)))
    mean_max = np.mean(pd.Series(bouts).apply(max_DFF, args=(cell, arr_dff)))
    output = std_max/mean_max
    return output


def dispersion_signal2(cell, filtered_dff):
    arr_dff = np.array(filtered_dff)
    std_max = np.std(pd.Series(bouts).apply(max_DFF, args=(cell, arr_dff)))
    mean_max = np.mean(pd.Series(bouts).apply(max_DFF, args=(cell, arr_dff)))
    return std_max, mean_max


def round_fps(fps_2p):
    # Filter DFF using the running average method, with different number of frames to average.
    # We start from 3 frames taken to average, and go up to the number of frames in one second (frame rate of the 2P).
    # Because apparently the function (in matlab) works better with uneven number of frames,
    # we would add +1 to this number if it is even.
    if int(fps_2p % 2) == 0:
        round_fps_beh = fps_2p + 1
        # Because the function only accepts integers, you need to round up the frame rate.
        # this is achieved with the function math.ceil
        if type(fps_2p) is float:
            # if frequency is not an integer, round it up.
            round_fps_beh = math.ceil(round_fps_beh)
    elif type(fps_2p) is float:
        round_fps_beh = math.ceil(fps_2p)
    else:
        round_fps_beh = fps_2p
    print('round max frames to average', round_fps_beh)
    return round_fps_beh


round_fps_beh = round_fps(fps_2p)
windows = list(range(3,round_fps_beh+1, 2))
print('window to try:', windows)
output_struct['Windows_filtering'] = windows

# define cells of interest as the most responding ones.
# these cells will be of use to assess if your filtering method works well.
# if no behavior is available, select random cells.
cells_interest = cells_most_responsive(4, bouts, DFF, cells_index, time_indices, experiment, analysis_log)
# define ylim for DFF plot as the min and max value of DFF across all cells.
dff_max, dff_min = np.nanmax(DFF), np.nanmin(DFF)

filtering_analysis(windows, cells_interest)
plt.show()
chosen_window = int(input('Chosen window for running average ?'))
print(chosen_window)

filtered_dff, filtered_noise = final_filtering(chosen_window)
arr_dff = np.array(filtered_dff)
output_struct['filtered_dff'] = filtered_dff
output_struct['filtered_noise'] = filtered_noise
output_struct['chosen_window'] = chosen_window

# Save config file and output structure
with open(output_path + 'logs/' + fishlabel + '_' + trial + '_analysis_log', 'wb') as fp:
    pickle.dump(analysis_log, fp)
    print('Analysis log was saved in', fp.name)
with open(output_path + 'dataset/' + fishlabel + '/' + trial + '/struct', 'wb') as output:
    pickle.dump(output_struct, output, protocol=pickle.HIGHEST_PROTOCOL)
    print('Output struct was saved in', output.name)

########################################################################################################

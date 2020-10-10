import time
import random
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import *
from utils.import_data import *


class ReAnalyze(Exception):
    pass


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def load_2p_output(fishlabel, trial, output_path):
    """
    Load every output that the suite2p gives you
    Arguments given are fishlabel, trial and folder_path.
    If folder_path is give,;
    Returns F, Fneu, spks, stat, ops, iscell
    """
    F = np.load(output_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/F.npy', allow_pickle=True)
    Fneu = np.load(output_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/Fneu.npy', allow_pickle=True)
    spks = np.load(output_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/spks.npy', allow_pickle=True)
    stat = np.load(output_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/stat.npy', allow_pickle=True)
    ops = np.load(output_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/ops.npy', allow_pickle=True)
    ops = ops.item()
    iscell = np.load(output_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/iscell.npy', allow_pickle=True)
    print('successfully loaded 2P data')
    return F, Fneu, spks, stat, ops, iscell


def correct_motion_artifact(F_corrected, cells_index, bad_frames):
    """
    Exclude from analysis the frames where there was motion artifact in the signal.

    Two steps of process:
    1) Automatic exclusion of frames using suite2p analysis parameters.
        Exclude frames for which registration was not very efficient and even after correction, there is still a low
        correlation in X and Y axis of this frame with the reference plane. These frames are refered to as bad_frames
    2) Manual exclusion of frames by visually selecting them.

    :param bad_frames is an array with 'bad_frames' indices.
    """
    # replace fluorescence value in frames where there was movement by 0
    F_corrected[:, bad_frames] = np.nan
    del_frames = list(bad_frames[0])
    print('frames to NaN based on automatic detection:', del_frames)
    time.sleep(2)
    print('Manual addition of frames to NaN')
    print('Click on bad frames on the next graph (left click)')
    print('Middle click to exit, right click to remove last point')
    time.sleep(2)
    # plot the fluorescence of 20 random cells and ask user to click on frames with motion artifact
    # user can skip this by middle click
    plt.figure('Motion artifact correction', figsize=[25, 15])
    random_cells = random.sample(list(cells_index), 20)
    plt.plot(np.transpose(F_corrected[random_cells]))
    plt.show()
    # get frames to delete by getting user input on the graph
    frames_ma = plt.ginput(n=-1, show_clicks=True)
    plt.close()
    # ask if user also want to manually add frame indices
    new_frames_ma = input('Additionnal frames to delete ? '
                              '(type frames number separated by space)'
                              '\n press enter if none')
    if frames_ma:
        for frame in frames_ma:
            x, y = frame
            F_corrected[:, int(x)] = np.nan
            del_frames.append(int(x))
    if new_frames_ma:
        new_frames_ma = [int(x) for x in new_frames_ma.split()]
        for frame in new_frames_ma:
            F_corrected[:, frame] = np.nan
            if frame not in del_frames:
                del_frames.append(frame)
    print('In the end, frames ', del_frames, 'were NaNed.')
    return F_corrected, del_frames


def correct_2p_outputs(fishlabel, trial, output_path, bad_frames, reanalyze, analysis_log, data_path):
    """
    Loads outputs from the 2psuite analysis of calcium signal

    Prints the original number of ROIs detected in this plane, the number of frames
    Prints the selected ROIs of interest (as number of cells)
    Correct the DFF with the neuropile activity and by substracting the median to shift the signal towards zero
    Returns the numpy array of the DFF corrected, and the list of indices of the cells

    """
    # Try to load corrected fluorescence and cells index if you already analyzed the files.
    try:
        F_corrected = np.load(output_path + 'np_array/' + fishlabel + '/' + trial + '/F_corrected_' +
                              trial + '.npy')
        cells_index = np.load(output_path + 'np_array/' + fishlabel + '/' + trial + '/cells_index_' +
                              trial + '.npy')
        del_frames = analysis_log['removed_frames']
        print('Loading existing F_corrected and cells index')
        if reanalyze:
            print('re analysing')
            raise ReAnalyze
    # IF not, correct it with the function 'correct_2p_outputs'
    except (FileNotFoundError, ReAnalyze):
        F, Fneu, spks, stat, ops, iscell = load_2p_output(fishlabel, trial, data_path)
        nROIs, nFrames = F.shape
        print('Number of ROIs: ', nROIs)
        print('Number of frames: ', nFrames)
        print(F.shape)
        cells_index = np.flatnonzero(iscell[:, 0])
        print(cells_index)
        nCells = len(cells_index)
        print('Number of cells: ', nCells)
        F_corrected = np.ones(F.shape)
        # correction based on recommandation of suite2p, correction made with the neuropile fluorescence
        for ROI in range(len(F)):
            F_corrected[ROI] = F[ROI] - 0.7 * Fneu[ROI]
        # if you don't want motion artifact correction automated, comment the next line
        print('Calculating F corrected and cells index')
        F_corrected, del_frames = correct_motion_artifact(F_corrected, cells_index, bad_frames)
        np.save(output_path + 'np_array/' + fishlabel + '/' + trial + '/F_corrected_' + trial + '.npy',
                F_corrected)
        np.save(output_path + 'np_array/' + fishlabel + '/' + trial + '/cells_index_' + trial + '.npy',
                cells_index)
    return F_corrected, cells_index, del_frames


def get_pos_x(cell_number, stat):
    """Middle position on the short axis"""
    return stat[cell_number]['med'][0]


def get_pos_y(cell_number, stat):
    """Returns the middle position of the cell masks on the long axis"""
    return stat[cell_number]['med'][1]


def create_TA_array(fishlabel, trial, path):
    corrected_frame_dataset, corrected_bout_dataset = load_behavior_dataframe(fishlabel, trial, path)
    tail_angle = corrected_frame_dataset.Tail_angle
    output_array = np.zeros((len(corrected_frame_dataset),2))
    output_array[:,1] = tail_angle
    output_array[:, 0] = corrected_frame_dataset.Time_index
    np.save(path + 'np_array/' + fishlabel + '/' + trial + '/TA_all' + str(trial), output_array)
    return output_array


def calc_DFF_reuse(F_corrected, cells_index, analysis_log):
    lim_inf, lim_sup = analysis_log['baseline_lim']

    DFF = np.ones(F_corrected.shape)
    for roi in cells_index:
        baseline = np.mean(F_corrected[roi][int(lim_inf):int(lim_sup)])
        DFF[roi] = (F_corrected[roi] - baseline) / baseline
    return DFF


def calc_DFF_click(F_corrected, cells_index, fps_2p, analysis_log):
    """ Calculates DFF based on fluorescence and a basal fluorescence.

    DFF equals F-F0/F0 where
    F is fluorescence at time step, and F0 basal fluorescence.
    Basal fluorescence is found by calculating the mean fluorescence value of a ROI during a window of time.
    Window of time is defined by the user, by selecting manually a period where ROIs were quiet.

    """
    print('Mouse input from matplotlib fig to define baseline')
    # if limits for baseline are not defined, ask user to choose them
    print('Manual selection of the baseline start and end. Left click on the graph to select start and end.')
    print('You can also click only once and end selection by middle click. Then the end of the baseline will be '
          'taken 10seconds after the beginning when possible.')
    print('You can remove previous point using right click.')
    time.sleep(3)
    plt.figure('Corrected fluorescence')
    plt.plot(np.transpose(F_corrected[cells_index]))
    plt.show()
    # when choosing it, you can do 2 clicks (lim inf and lim sup) or one (lim inf)
    # if only one is given, it will take the superior limit 10 seconds after the first one.
    baseline = plt.ginput(n=2, show_clicks=True)
    plt.close()
    lim_inf, a = baseline[0]
    if len(baseline) == 1:
        lim_sup = lim_inf + 10*fps_2p
    elif len(baseline) == 2:
        lim_sup, a = baseline[1]
    lim_inf = int(lim_inf)
    lim_sup = int(lim_sup)
    print('You chose lim inf:', lim_inf)
    print('lim sup:', lim_sup)
    analysis_log['baseline_lim'] = (lim_inf, lim_sup)

    DFF = np.ones(F_corrected.shape)
    for roi in cells_index:
        baseline = np.mean(F_corrected[roi][int(lim_inf):int(lim_sup)])
        DFF[roi] = (F_corrected[roi] - baseline) / baseline
    return DFF, analysis_log


def calc_DFF_manual(F_corrected, cells_index, analysis_log):
    """
    Calculates DFF based on fluorescence and a basal fluorescence.

    DFF equals F-F0/F0 where
    F is fluorescence at time step, and F0 basal fluorescence.
    Basal fluorescence is found by calculating the mean fluorescence value of a ROI during a window of time.
    Window of time is defined by the user, by selecting manually a period where ROIs were quiet.

    """
    print('Manual input to define baseline')

    lim_inf = int(input('Inferior limit of F_0? (index of frame) '))
    lim_sup = int(input('Superior limit of F_0? (index of frame) '))
    analysis_log['baseline_lim'] = (lim_inf, lim_sup)

    DFF = np.ones(F_corrected.shape)
    for roi in cells_index:
        baseline = np.mean(F_corrected[roi][int(lim_inf):int(lim_sup)])
        DFF[roi] = (F_corrected[roi] - baseline) / baseline
    return DFF, analysis_log


def calc_noise(DFF, cells_index, analysis_log):
    noise = np.zeros(DFF.shape[0])
    F0_inf, F0_sup = analysis_log['baseline_lim']
    for roi in cells_index:
        noise[roi] = np.std(DFF[roi, F0_inf:F0_sup])
    return noise


def max_DFF(bout, cell, DFF, time_indices, experiment, analysis_log):
    """For a given bout and a given cell, find the maximum of DFF value reached by the cell
    between start of the bout and +x second after end of bout
    Time window x can be set by the user (see initialisation of the config_file)
    """
    start = (bout.start / experiment.fps_beh)
    ca_indices_frame = find_indices(time_indices, lambda e: start < e < start + analysis_log['window_max_dff'])
    DFF_bout_only = DFF[cell, ca_indices_frame]
    try:
        output = np.nanmax(DFF_bout_only)
    except ValueError:
        output = np.nan
    return output


def max_DFF_cell(cell, bouts, DFF, time_indices, experiment, analysis_log):
    col = pd.Series(bouts).apply(max_DFF, args=(cell, DFF, time_indices, experiment, analysis_log))
    return col


def max_DFF_bout(bout, bouts, DFF, cells_index, time_indices, experiment, analysis_log):
    col = pd.Series(cells_index)


def cells_most_responsive(n, bouts, DFF, cells_index, time_indices, experiment, analysis_log):
    """"
    Returns the n cells with highest max DF/F of all cells.

    :param n is an int of the number of cells you want to sort.
    :param DFF is an array of DF/F value for each cell (line) at each time point (column).

    :return n cells with highest max DF/F.

    """
    max_dff = [0] * len(cells_index)
    for i, cell in enumerate(cells_index):
        max_dff[i] = np.nanmax(pd.Series(bouts).apply(max_DFF, args=(cell, DFF, time_indices, experiment, analysis_log)))

    max_dff_sorted = max_dff.copy()
    max_dff_sorted.sort()
    a = max_dff_sorted[-n*2:-n]
    indices = find_indices(max_dff, lambda e: e in a)
    most_resp = [cells_index[i] for i in indices]
    return most_resp


def load_behavior(fishlabel, trial, path):
    # if behavior is not analyzed, load a false TA array to still use the same function
    try:
        # load .txt file from ZZ output
        df_bouts = pd.read_pickle(path + 'dataset/' + fishlabel +
                                  '/' + fishlabel + '_analyzed_bout_dataset_' + trial)
        TA_all = create_TA_array(fishlabel, trial, path)
        print('Loading automatically tracked behavior')
    except FileNotFoundError:
        print('no automatic tracked behavior (ZZ)')
    return df_bouts, TA_all


def correct_iscell(fishlabel, trial, x_lim, y_lim, folder_path=False):
    print('Correcting ROIs automatically identified as cells in fish', fishlabel, 'trial', trial)
    print('ROIs which are localised outside of x range', x_lim, 'will be discarded')
    print('ROIs which are localised outside of y range', y_lim, 'will be discarded')
    F, Fneu, spks, stat, ops, iscell = load_2p_output(fishlabel, trial, folder_path)
    iscell_corrected = iscell.copy()
    roi_excluded = []
    cells_index = np.flatnonzero(iscell[:, 0])
    for cell in cells_index:
        if (x_lim[1] <= get_pos_x(cell, stat)) or (get_pos_x(cell, stat) <= x_lim[0]):
            iscell_corrected[cell, 0] = 0
            roi_excluded.append(cell)
        elif (y_lim[1] <= get_pos_y(cell, stat)) or (get_pos_y(cell, stat) <= y_lim[0]):
            iscell_corrected[cell, 0] = 0
            roi_excluded.append(cell)
    print('number of cells initial: ', len(cells_index))
    print('number of cells excluded:', len(roi_excluded))
    print('excluded: ', roi_excluded)
    cells_index_corrected = np.flatnonzero(iscell_corrected[:, 0])
    print('number of rois kept:', len(cells_index_corrected))
    save = str(input('DO you want to save new iscell file ? (yes)/no'))
    if not save == 'no':
        np.save(folder_path + '/' + trial + '/suite2p/plane0' + '/iscell.npy', iscell_corrected)
        print('new iscell file is saved in this directory:', folder_path + '/' + trial + '/suite2p/plane0' )
    return iscell_corrected


def separate_cell_regions(cells_index, pont_mid, mid_bulb, stat):
    pontine_cells = []
    middle_cells = []
    bulbar_cells = []
    for cell in cells_index:
        if get_pos_y(cell, stat) >= pont_mid:
            pontine_cells.append(cell)
        elif mid_bulb < get_pos_y(cell, stat) < pont_mid :
            middle_cells.append(cell)
        else:
            bulbar_cells.append(cell)
    return pontine_cells, middle_cells, bulbar_cells


def run_avg_filter(DFF, noise, window, cells_index, analysis_log):
    filtered_dff = DFF.copy()
    filtered_noise = np.zeros(noise.shape)

    filtered_dff = pd.DataFrame(filtered_dff).rolling(window=window, center=True, axis=1).mean()

    lim_inf, lim_sup = analysis_log['baseline_lim']
    for ROI in cells_index:
        filtered_noise[ROI] = np.std(filtered_dff.iloc[ROI, lim_inf:lim_sup])
    return filtered_dff, filtered_noise


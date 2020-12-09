import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as sns

from utils.functions_calcium_analysis import *
from utils.plotting import *


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


fishlabel = '200813_F1'
depth = '160um'
# output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'
# data_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/'
#
# experiment = load_experiment(output_path, fishlabel)[0]
# fps_2p = experiment.fps_2p
# fps_beh = experiment.fps_beh
#
# F, Fneu, spks, stat, ops, iscell = load_suite2p_outputs(fishlabel, depth, data_path + 'suite2p_output/')
# # load output struct if exisiting
# struct = load_output_struct(fishlabel, depth, output_path + 'dataset/', data_path)
#
# nFrames, nROIs = struct['nFrames'], struct['nROI']
# time_indices = struct['time_indices']
# create time points to plot everything according to time and not frame index.
# time_indices = [frame*(1/fps_2p) for frame in range(nFrames)]

# group_csv = pd.read_csv(data_path + fishlabel + '/' + depth + 'group_pos.csv')

with open("/Users/test/Google Drive/PhD/Data/200813_F1/160um/struct", 'rb') as handle:
    struct = pickle.load(handle)

cells = np.flatnonzero(struct['iscell'][:, 0])

bulbar_lat_lims = [55, 123]
pontine_bulbar_lim = 220
midline = 92
bulbar_lat = []
bulbar_medial = []
pontine = []

cells_group = [np.nan] * len(struct['iscell'])
side = [np.nan] * len(struct['iscell'])

for cell in cells:
    if (struct['stat'][cell]['med'][0] <= bulbar_lat_lims[0]) or (struct['stat'][cell]['med'][0] >= bulbar_lat_lims[1]):
        bulbar_lat.append(cell)
        cells_group[cell] = 'bulbar_lateral'
    elif struct['stat'][cell]['med'][1] > pontine_bulbar_lim:
        bulbar_medial.append(cell)
        cells_group[cell] = 'bulbar_medial'
    elif struct['stat'][cell]['med'][1] < pontine_bulbar_lim:
        pontine.append(cell)
        cells_group[cell] = 'pontine'

    if struct['stat'][cell]['med'][0] < midline:
        side[cell] = 'right'
    else:
        side[cell] = 'left'

# TODO: add another group to separate bulbar medial rostral and caudal

print('Number of pontine cells:', len(pontine))
print('Number of bulbar medial cells', len(bulbar_medial))
print('Number of bulbar lateral cells', len(bulbar_lat))

colors = {'pontine': 'blue',
          'bulbar_medial': 'orange',
          'bulbar_lateral': 'green'}

colors_int = {'pontine': 0,
              'bulbar_medial': 1,
              'bulbar_lateral': 2}

plt.figure()
plt.imshow(struct['ops']['meanImg'])

for cell in cells:
    plt.plot(struct['stat'][cell]['med'][1],
             struct['stat'][cell]['med'][0],
             'o',
             color=colors[cells_group[cell]],
             label='cell ' + str(cell))
plt.title('Cells group for fish ' + fishlabel + ', depth:' + depth)
plt.ylabel('LEFT / RIGHT')
plt.xlabel('ROSTRAL / CAUDAL')
plt.savefig('/network/lustre/iss01/wyart/analyses/mathilde.lapoix/HE_behavior_CI_analysis/' +
            fishlabel + '_' + depth + '_cellPoseGroup_median.png')
plt.show()

plt.figure()
heatmap_group = np.zeros((struct['ops']['Ly'], struct['ops']['Lx']))
heatmap_group[:] = np.nan
for cell in cells:
    ypix = struct['stat'][cell]['ypix']
    xpix = struct['stat'][cell]['xpix']
    heatmap_group[ypix, xpix] = colors_int[cells_group[cell]]
plt.imshow(struct['ops']['meanImg'], cmap='Greys_r')
plt.imshow(heatmap_group, alpha=0.85, cmap='tab10', vmax=4)
plt.plot([pontine_bulbar_lim, pontine_bulbar_lim], [0, struct['ops']['meanImg'].shape[0] - 1], '--')
plt.title('Cells group for fish ' + fishlabel + ', depth:' + depth)
plt.ylabel('LEFT / RIGHT')
plt.xlabel('ROSTRAL / CAUDAL')
plt.savefig('/network/lustre/iss01/wyart/analyses/mathilde.lapoix/HE_behavior_CI_analysis/' +
            fishlabel + '_' + depth + '_cellPoseGroup_mask.png')
plt.figure()
heatmap_side = np.zeros((struct['ops']['Ly'], struct['ops']['Lx']))
heatmap_side[:] = np.nan
for cell in cells:
    ypix = struct['stat'][cell]['ypix']
    xpix = struct['stat'][cell]['xpix']
    side_cell = side[cell]
    if side_cell == 'right':
        heatmap_side[ypix, xpix] = 0
    elif side_cell == 'left':
        heatmap_side[ypix, xpix] = 1
plt.imshow(struct['ops']['meanImg'], cmap='Greys_r')
plt.imshow(heatmap_side, alpha=0.85, cmap='PiYG')
plt.plot([0, struct['ops']['meanImg'].shape[1] - 1], [midline] * 2, '--')
plt.title('Cells side for fish ' + fishlabel + ', depth:' + depth)
plt.ylabel('LEFT / RIGHT')
plt.xlabel('ROSTRAL / CAUDAL')
plt.savefig('/network/lustre/iss01/wyart/analyses/mathilde.lapoix/HE_behavior_CI_analysis/' +
            fishlabel + '_' + depth + '_cellPoseSide_mask.png')

###########################################################

# Create dataframe long format

df_bouts = struct['df_bouts']
nBouts = len(struct['df_bouts'])
nFrames = struct['DFF'].shape[1]
frame_rate = 4.4
frame_rate_beh = 300
time_indices = [frame * (1 / frame_rate) for frame in range(nFrames)]
lim_sup = 0.2


def get_pos_x(cell_number, stat):
    """Middle position on the short axis"""
    return stat[cell_number]['med'][0]


def get_pos_y(cell_number, stat):
    """Returns the middle position of the cell masks on the long axis"""
    return stat[cell_number]['med'][1]


x_positions = pd.Series(cells).apply(get_pos_x, args=(struct['stat'],))
y_positions = pd.Series(cells).apply(get_pos_y, args=(struct['stat'],))
side = np.array(side)
cells_group = np.array(cells_group)

df_lf = pd.DataFrame({'cell': np.repeat(cells, nBouts),
                      'bout': np.tile(list(range(nBouts)), len(cells)),
                      'bout_type': np.tile(struct['df_bouts'].category, len(cells)),
                      'bout_dur': np.tile(struct['df_bouts'].Bout_Duration, len(cells)),
                      'n_osc': np.tile(struct['df_bouts'].Number_Osc, len(cells)),
                      'abs_Max_Bend_Amp': np.tile(struct['df_bouts'].abs_Max_Bend_Amp, len(cells)),
                      'Integral_TA': np.tile(struct['df_bouts'].Integral_TA, len(cells)),
                      'x_pos': np.repeat(x_positions, nBouts),
                      'y_pos': np.repeat(y_positions, nBouts),
                      'side': np.repeat(side[cells], nBouts),
                      'group': np.repeat(cells_group[cells], nBouts),
                      'plane': [depth] * len(np.repeat(cells, nBouts)),
                      'max_dff_f': [0] * len(np.repeat(cells, nBouts)),
                      'index_max_dff_f': [0] * len(np.repeat(cells, nBouts)),
                      'recruitment_f': [0] * len(np.repeat(cells, nBouts))

                      })


# TODO: change function to find max start - 1 or 2 frames
def find_max_index(bout, roi, df_bouts, dff, frame_rate, time_indices, lim_sup):
    """Get the frame index at which the max_dff was found around a bout"""
    start = df_bouts['BoutStart_summed'].iloc[bout] / frame_rate
    end = df_bouts['BoutEnd_summed'].iloc[bout] / frame_rate
    ca_indices_frame = find_indices(time_indices, lambda e: start - 1 < e < end + lim_sup)
    DFF_bout_only = dff[roi, ca_indices_frame]
    try:
        output = ca_indices_frame[0] + np.nanargmax(DFF_bout_only)
    except ValueError:
        output = np.nan
    return output


def calc_max_dff(bout, roi, df_bouts, dff, frame_rate, time_indices, lim_sup):
    start = df_bouts['BoutStart_summed'].iloc[bout] / frame_rate
    end = df_bouts['BoutEnd_summed'].iloc[bout] / frame_rate
    ca_indices_frame = find_indices(time_indices, lambda e: start - 1 < e < end + lim_sup)
    DFF_bout_only = dff[roi, ca_indices_frame]
    try:
        output = np.nanmax(DFF_bout_only)
    except ValueError:
        output = np.nan
    return output


def calc_cell_max_all_bout(roi, df_bouts, dff, frame_rate, time_indices, lim_sup):
    """Calculate, for one cell, the max dff reached by this cell around each boutulation.
    The signal in which the max dff will be looked for is defined by windows."""

    return pd.Series(range(nBouts)).apply(calc_max_dff,
                                          args=(roi, df_bouts, dff, frame_rate, time_indices, lim_sup))


def calc_cell_max_frame_all_bout(roi, df_bouts, dff, frame_rate, time_indices, lim_sup):
    """Calculate, for one cell, the index at which the max dff was reached around each boutulation.
    The signal in which the max dff will be looked for is defined by windows.
    See doc of find_max_index for more information."""

    return pd.Series(range(nBouts)).apply(find_max_index,
                                          args=(roi, df_bouts, dff, frame_rate, time_indices, lim_sup))


# fill max DF/F filtered column
maxs_filtered = pd.Series(cells).apply(calc_cell_max_all_bout,
                                       args=(struct['df_bouts'], struct['DFF_filtered'],
                                             frame_rate_beh, struct['time_indices'], lim_sup))
df_lf.at[:, 'max_dff_f'] = np.array(maxs_filtered).flatten()

max_indices_f = pd.Series(cells).apply(calc_cell_max_frame_all_bout,
                                       args=(struct['df_bouts'], struct['DFF_filtered'],
                                             frame_rate_beh, struct['time_indices'], lim_sup))
df_lf.at[:, 'index_max_dff_f'] = np.array(max_indices_f).flatten()


# Create DataFrame with signal


def build_df_signal_bout(df_bouts, lim_sup, time_indices, frame_rate_beh):
    df = pd.DataFrame(columns=['bout',
                               'bout_type', 'bout_dur', 'n_osc', 'abs_Max_Bend_Amp', 'Integral_TA',
                               'relative_time_point',
                               'plane'],
                      index=np.arange(nFrames))
    for bout in np.arange(nBouts):
        start = df_bouts['BoutStart_summed'].iloc[bout]/frame_rate_beh
        end = df_bouts['BoutStart_summed'].iloc[bout]/frame_rate_beh
        ca_indices_frame = find_indices(time_indices, lambda e: start < e < end + lim_sup)

        ci_start = ca_indices_frame[0] - 2  # take 2 frames before the actual start
        if len(ca_indices_frame) > 1:
            ci_end = ca_indices_frame[-1]
        else:
            ci_end = ca_indices_frame[0]

        df['bout'].iloc[ci_start:ci_end] = bout
        df['bout_type'].iloc[ci_start:ci_end] = df_bouts['category'].iloc[bout]
        df['bout_dur'].iloc[ci_start:ci_end] = df_bouts['Bout_Duration'].iloc[bout]
        df['n_osc'].iloc[ci_start:ci_end] = df_bouts['Number_Osc'].iloc[bout]
        df['abs_Max_Bend_Amp'].iloc[ci_start:ci_end] = df_bouts['abs_Max_Bend_Amp'].iloc[bout]
        df['Integral_TA'].iloc[ci_start:ci_end] = df_bouts['Integral_TA'].iloc[bout]

        # find closest frame to bout onset
        possible_frame_start = [ca_indices_frame[0], ca_indices_frame[0]-1, ca_indices_frame[0]+1]
        real_start = possible_frame_start[np.argmin([abs(i*frame_rate - start) for i in possible_frame_start])]
        relative_time_points = np.arange(int(real_start-2*frame_rate), int(real_start+2*frame_rate))
        df['relative_time_point'].iloc[relative_time_points] = np.arange(0, len(relative_time_points))
    return df


df_signal_bout = build_df_signal_bout(struct['df_bouts'], lim_sup, time_indices, frame_rate_beh)


def build_df_signal(roi, dff, dff_filtered, cells_group, df_signal_bout):
    df = pd.DataFrame(columns=['cell', 'bout', 'time_point', 'dff', 'dff_f',
                               'bout_type', 'bout_dur', 'n_osc', 'abs_Max_Bend_Amp', 'Integral_TA',
                               'relative_time_point',
                               'x_pos', 'y_pos', 'cluster', 'group', 'side',
                               'plane'],
                      index=np.arange(nFrames))
    df['cell'] = [roi] * nFrames
    df['x_pos']: np.repeat(x_positions, nFrames)
    df['y_pos']: np.repeat(y_positions, nFrames)
    df['time_point'] = time_indices
    df['dff'] = dff[roi]
    df['dff_f'] = dff_filtered[roi]
    df['group'] = [cells_group[roi]] * nFrames
    df['side'] = [side[roi]] * nFrames
    df['plane'] = [depth] * nFrames

    df['bout'] = df_signal_bout.bout
    df['bout_type'] = df_signal_bout.bout_type
    df['bout_dur'] = df_signal_bout.bout_dur
    df['n_osc'] = df_signal_bout.n_osc
    df['abs_Max_Bend_Amp'] = df_signal_bout.abs_Max_Bend_Amp
    df['Integral_TA'] = df_signal_bout.Integral_TA
    df['relative_time_point'] = df_signal_bout.relative_time_point

    return df


dict_df_signal = {}
for cell in cells:
    dict_df_signal[str(cell)] = build_df_signal(cell, struct['DFF'], struct['DFF_filtered'],
                                                cells_group, df_signal_bout)

df_signal_final = pd.concat(dict_df_signal)

sns.lineplot(data=df_signal_final[df_signal_final['bout_type'] == 'S'], x='relative_time_point', y='dff_f', hue='group')
plt.plot([frame_rate]*2, [0, 2], '--', 'r')

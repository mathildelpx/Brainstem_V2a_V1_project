import os, pickle, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from matplotlib import cm
import sys
import shelve
import seaborn as sns

from utils.import_data import *
from utils.createBoutClass import create_bout_objects
from NeuroFrance.utils.get_ci_vs_bh import *
from NeuroFrance.utils.groupsCells import get_cells_group
from NeuroFrance.utils.calcium_traces import get_pos_y, get_pos_x


csv_path = '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/chx10_RSN/summaryData_bulbar_group_only.csv'

fishlabel = '200813_F1'
plane = '180um'

summary = pd.read_csv(csv_path)
fish_mask = summary.fishlabel == fishlabel
plane_mask = summary.plane == plane
suite2p_path = list(summary.loc[fish_mask & plane_mask, 'suite2p_path'])[0]
# ZZ_path = list(summary.loc[fish_mask & plane_mask, 'ZZ_path'])[0]
output_path = list(summary.loc[fish_mask & plane_mask, 'output_path'])[0]
fps_ci = float(summary.loc[fish_mask & plane_mask, 'frameRate'])
fps_beh = float(summary.loc[fish_mask & plane_mask, 'frameRateBeh'])
nFrames = float(summary.loc[fish_mask & plane_mask, 'nFrames_2p'])

shelf_in = shelve.open(output_path + '/shelve_calciumAnalysis.out')
for key in shelf_in:
    globals()[key]=shelf_in[key]
shelf_in.close()

tail_angle = np.load(output_path+'/dataset/tail_angle.npy')
df_bout = pd.read_pickle(output_path +'/dataset/df_bout')
df_frame = pd.read_pickle(output_path+'/dataset/df_frame')
# cells group
side_lim = summary.loc[fish_mask & plane_mask, 'midline'].item()
cells_group, side, bulbar_lateral, bulbar_medial, pontine, spinal_cord = get_cells_group(summary, fishlabel, plane,
                                                                                         F, cells, stat)

cells_x_pos = np.array(pd.Series(cells).apply(get_pos_x, args=(stat,)))
cells_y_pos = np.array(pd.Series(cells).apply(get_pos_y, args=(stat,)))

# Load missing data

ta_resampled = np.load(output_path + '/dataset/tail_angle_resampled.npy')
binary_bh = np.load(output_path +'/dataset/tail_angle_binary.npy')
dff_filtered = dff_f_avg_inter.copy()

# Create df to store output
#

df = pd.DataFrame({'fishlabel': [fishlabel] * len(cells) * dff.shape[1],
                   'plane': [plane] * len(cells) * dff.shape[1],
                   'roi': np.repeat(cells, dff.shape[1]),
                   'frame': np.tile(np.arange(dff.shape[1]), len(cells)),
                   'dff': dff_filtered[cells, :].flatten(),
                   'dif_dff': [np.nan] * len(cells) * dff.shape[1],
                   'frame_state': [np.nan] * len(cells) * dff.shape[1],
                   'bout_type': [np.nan] * len(cells) * dff.shape[1],
                   'max_ta': [np.nan] * len(cells) * dff.shape[1],
                   'max_ta_bout': [np.nan] * len(cells) * dff.shape[1],
                   'duration_bout': [np.nan] * len(cells) * dff.shape[1],
                   'roi_group': np.repeat(cells_group[cells], dff.shape[1]),
                   'roi_side': np.repeat(side[cells],
                                         dff.shape[1]),
                   'roi_x_pos': np.repeat(cells_x_pos, dff.shape[1]),
                   'roi_y_pos': np.repeat(cells_y_pos, dff.shape[1])})

# Remove frames without behavior
#
to_keep = np.where(binary_bh != 0)[0]
print(len(to_keep))
df = df[df.frame.isin(to_keep)]
print(df.shape)

#
# FIRST: easy path of looking only at bout types and bout duration

df['bout_type'] = pd.Series(df.frame).apply(get_bout_type, args=(df_bout, df_frame, fps_ci, fps_beh))
df['max_ta_bout'] = pd.Series(df.frame).apply(get_bout_amp, args=(df_bout, df_frame, fps_ci, fps_beh))
df['duration_bout'] = pd.Series(df.frame).apply(get_bout_duration, args=(df_bout, df_frame, fps_ci, fps_beh))
df['tail_angle'] = np.tile(ta_resampled[to_keep], len(cells))

for roi in cells:
    df.loc[df.roi == roi, 'dif_dff'] = df.frame.apply(get_dif_dff, args=(roi, dff_filtered))

fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
for i, group in enumerate(['bulbar_lateral_right', 'bulbar_lateral_left', 'bulbar_medial']):
    sns.scatterplot(data=df[df.roi_group == group],
                    x='tail_angle', y='dff', ax=ax[i])
    ax[i].set_title(group)
    ax[i].set_xlabel('Tail angle [°] (right/left)')


fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
for i, group in enumerate(['bulbar_lateral_right', 'bulbar_lateral_left', 'bulbar_medial']):
    sns.scatterplot(data=df[df.roi_group == group],
                    x='max_ta_bout', y='dff', ax=ax[i])
    ax[i].set_title(group)
    ax[i].set_xlabel('Tail angle [°] (right/left)')


df['syl_type'] = df.bout_type.copy()
df['syl_type'] = df.frame.apply(get_syl_type, args=(df_frame, fps_ci, fps_beh))

#  get max amp of each syllabus
df['max_ta'] = df.frame.apply(get_syl_amp, args=(df_frame, fps_ci, fps_beh))

# get accumulated duration
df['accumulated_dur'] = np.nan

for frame in df.frame:

    if frame - 1 in df.frame:  # if movement during previous frame

        # if previous frame contained same bout type
        if set(df.loc[df.frame == frame - 1, 'syl_type']) == set(df.loc[df.frame == frame, 'syl_type']):

            # get previous accumulated duration
            previous_acc_dur = list(set(df.loc[df.frame == frame - 1, 'accumulated_dur']))[0]

            # put in dataframe
            df.loc[df.frame == frame, 'accumulated_dur'] = [round(previous_acc_dur + 1 / fps_ci, 4)] * len(cells)

        else:  # if not, beginning of the bout

            df.loc[df.frame == frame, 'accumulated_dur'] = [round(1 / fps_ci, 4)] * len(cells)

    else:  # if no mov before its the start of the bout

        df.loc[df.frame == frame, 'accumulated_dur'] = [round(1 / fps_ci, 4)] * len(cells)

# Build syllabus

## chunck behavior into syllabus

syl_num = -1
df['syl_num'] = np.nan
df_short = df[df.roi == 0]  # temporary

for frame in list(set(df_short.frame)):

    acc_dur = df_short.loc[df_short.frame == frame, 'accumulated_dur'].item()  # get accumulated dur of ongoing beh
    if acc_dur == np.nanmin(df_short['accumulated_dur']):  # if accumulated duration is equals to one frame
        syl_num += 1  # new syllabus started
    df.loc[df.frame == frame, 'syl_num'] = [syl_num] * len(cells)  # for this frame, enter the number of ongoing syl

## build dataframe for each syllabus

df_syllabus = pd.DataFrame({'fishlabel': [fishlabel] * syl_num,
                            'plane': [plane] * syl_num,
                            'syl': np.arange(syl_num),
                            'type': [np.nan] * syl_num,
                            'duration': [np.nan] * syl_num,
                            'start': [np.nan] * syl_num,
                            'end': [np.nan] * syl_num,
                            'max_ta': [np.nan] * syl_num,
                            'side': [np.nan] * syl_num,
                            'isolated': [1] * syl_num})

previous_end = 0
print(df_syllabus.describe())

for syl in np.arange(syl_num):

    typ = list(df.loc[df.syl_num == syl, 'syl_type'])[0]
    df_syllabus.loc[df_syllabus.syl == syl, 'type'] = typ

    start = list(df.loc[df.syl_num == syl, 'frame'])[0]
    end = list(df.loc[df.syl_num == syl, 'frame'])[-1]
    df_syllabus.loc[df_syllabus.syl == syl, 'start'] = start
    df_syllabus.loc[df_syllabus.syl == syl, 'end'] = end

    df_syllabus.loc[df_syllabus.syl == syl, 'duration'] = (end - start) / fps_ci

    ta = df_frame.Tail_angle[int((start / fps_ci) * fps_beh):int((end / fps_ci) * fps_beh)]
    try:
        df_syllabus.loc[df_syllabus.syl == syl, 'max_ta'] = max(np.nanmin(ta), np.nanmax(ta), key=abs)
    except ValueError:
        df_syllabus.loc[df_syllabus.syl == syl, 'max_ta'] = np.nan

    # check if syllabus is isolated
    time_to_previous = (start - previous_end)/fps_ci
    try:
        time_to_next = (list(df.loc[df.syl_num == syl+1, 'frame'])[0] - end)/fps_ci
    except KeyError:
        time_to_next = np.nan

    if (time_to_previous <= 0.4) | (time_to_next <= 0.4):
        df_syllabus.loc[df_syllabus.syl == syl, 'isolated'] = 0

    previous_end = end


df_syllabus['side'] = pd.Series(df_syllabus.syl).apply(get_syl_side, args=(df_syllabus,))

print(df_syllabus.describe())
print('Number of syllabus that were NaNed:', len(df_syllabus[df_syllabus.type.isna()]))
print('Number of isolated syllabus:', len(df_syllabus[df_syllabus.isolated == 1]),
      '\n', list(df_syllabus[df_syllabus.isolated == 1]['type']))

#   Build dataset with recruitment of roi for each syllabus

df_syllabus_roi = pd.DataFrame({'fishlabel': [fishlabel] * len(cells) * len(df_syllabus),
                                'plane': [plane] * len(cells) * len(df_syllabus),
                                'syl': np.tile(df_syllabus.syl, len(cells)),
                                'roi': np.repeat(cells, len(df_syllabus)),
                                'type': np.tile(df_syllabus.type, len(cells)),
                                'duration': np.tile(df_syllabus.duration, len(cells)),
                                'start': np.tile(df_syllabus.start, len(cells)),
                                'end': np.tile(df_syllabus.end, len(cells)),
                                'max_ta': np.tile(df_syllabus.max_ta, len(cells)),
                                'syl_side': np.tile(df_syllabus.side, len(cells)),
                                'syl_isolated': np.tile(df_syllabus.isolated, len(cells)),
                                'max_dff': [np.nan] * len(cells) * len(df_syllabus),
                                'max_dff_norm': [np.nan] * len(cells) * len(df_syllabus),
                                'dif_dff_start_end': [np.nan] * len(cells) * len(df_syllabus),
                                'recruitment': [np.nan] * len(cells) * len(df_syllabus),
                                'roi_group': np.repeat(cells_group[cells],
                                                       len(df_syllabus)),
                                'roi_side': np.repeat(side[cells],
                                                      len(df_syllabus)),
                                'roi_x_pos': np.repeat(cells_x_pos,
                                                       len(df_syllabus)),
                                'roi_y_pos': np.repeat(cells_y_pos,
                                                       len(df_syllabus))})

for roi in cells:
    df_syllabus_roi.loc[df_syllabus_roi.roi == roi, 'dif_dff_start_end'] = list(pd.Series(df_syllabus.syl).apply(
        get_dif_start_end, args=(roi, df_syllabus, dff_filtered)))
    df_syllabus_roi.loc[df_syllabus_roi.roi == roi, 'max_dff'] = list(pd.Series(df_syllabus.syl).apply(
        get_max_dff, args=(roi, df_syllabus, dff_filtered)))
    df_syllabus_roi.loc[df_syllabus_roi.roi == roi, 'max_dff_norm'] = list(pd.Series(df_syllabus.syl).apply(
        get_max_dff_norm, args=(roi, df_syllabus, dff_filtered)))

#  Save datasets
df.to_pickle(output_path + '/dataset/df_frame_roi')
df_syllabus.to_pickle(output_path + '/dataset/df_syllabus')
df_syllabus_roi.to_pickle((output_path + '/dataset/df_syllabus_roi'))

#  Plot Decomposition of bouts into syllabus

plt.figure()
plt.plot(np.array(df_frame.Time_index), np.array(df_frame.Tail_angle), color='black')
for syl in df_syllabus.index:
    start, end = int(df_syllabus['start'].iloc[syl]), int(df_syllabus['end'].iloc[syl])
    plt.plot([time_trace_f[start], time_trace_f[end]], [0,0], 'o')
plt.savefig(output_path + '/fig/syllabus_decompose.svg')

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
for i, group in enumerate(['bulbar_lateral_right', 'bulbar_lateral_left', 'bulbar_medial']):
    sns.scatterplot(data=df_syllabus_roi[df_syllabus_roi.roi_group == group],
                    x='max_ta', y='dif_dff_start_end', ax=ax[i])
    ax[i].set_title(group)
    ax[i].set_xlabel('Tail angle [°] (right/left)')

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
for i, group in enumerate(['bulbar_lateral_right', 'bulbar_lateral_left', 'bulbar_medial']):
    sns.scatterplot(data=df_syllabus_roi[df_syllabus_roi.roi_group == group],
                    x='max_ta', y='max_dff', ax=ax[i])
    ax[i].set_title(group)
    ax[i].set_xlabel('Tail angle [°] (right/left)')
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
for i, group in enumerate(['bulbar_lateral_right', 'bulbar_lateral_left', 'bulbar_medial']):
    sns.scatterplot(data=df_syllabus_roi[df_syllabus_roi.roi_group == group],
                    x='max_ta', y='max_dff_norm', ax=ax[i])
    ax[i].set_title(group)
    ax[i].set_xlabel('Tail angle [°] (right/left)')

#  Plot heatmap of mean recruitment during the different syllabus type, for each population

fig, ax = plt.subplots(3, 3, figsize=(14, 14))

for i, param in enumerate(['max_dff', 'dif_dff_start_end', 'max_dff_norm']):

    a, b = ops['meanImg'].shape
    heatmaps = np.zeros((a, b, 3))

    for j, syl_type in enumerate(['F', 'L', 'R']):
        heatmap = np.zeros(ops['meanImg'].shape)
        heatmap[:] = np.nan

        for cell in cells:
            value = np.nanmedian(
                df_syllabus_roi.loc[(df_syllabus_roi.roi == cell) & (df_syllabus_roi.type == syl_type), param])
            xpix, ypix = stat[cell]['xpix'], stat[cell]['ypix']
            heatmap[ypix, xpix] = value
        ax[i, j].imshow(ops['meanImg'], cmap='Greys')
        count_events = len(df_syllabus_roi[(df_syllabus_roi.roi == cell) & (df_syllabus_roi.type == syl_type)])
        ax[i, j].set_title('Median {} of cells during {} syllabus. ({} events)'.format(param, side, count_events))
        heatmaps[:, :, j] = heatmap

    vmin = np.nanmin(heatmaps)
    vmax = np.nanmax(heatmaps)

    for j, side in enumerate(['F', 'L', 'R']):
        im = ax[i, j].imshow(heatmaps[:, :, j], cmap='plasma', vmin=vmin, vmax=vmax)

    plt.colorbar(im, ax=ax[i].ravel().tolist())


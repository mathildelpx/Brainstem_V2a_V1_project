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
from NeuroFrance.utils.regressor_tools import create_regressor, pearson_coef


csv_path = '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/chx10_RSN/summaryData_bulbar_group_only.csv'

fishlabel = '200930_F1'
plane = '96um'

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

F, Fneu, spks, stat, ops, iscell = load_suite2p_outputs('', '/', suite2p_path)
time_trace_f = np.arange(F.shape[1])/fps_ci
del suite2p_path

tail_angle = np.load(output_path+'/dataset/tail_angle.npy')

# Get cells group and pos

side_lim = summary.loc[fish_mask & plane_mask, 'midline'].item()
cells_group, side, bulbar_lateral, bulbar_medial, pontine, spinal_cord = get_cells_group(summary, fishlabel, plane,
                                                                                         F, cells, stat)

cells_x_pos = np.array(pd.Series(cells).apply(get_pos_x, args=(stat,)))
cells_y_pos = np.array(pd.Series(cells).apply(get_pos_y, args=(stat,)))

# Build DataFrame

df = pd.DataFrame({'fishlabel': [fishlabel]*len(cells),
                   'plane': [plane]*len(cells),
                   'cell': cells,
                   'group': np.nan,
                   'x_pos': cells_x_pos,
                   'y_pos': cells_y_pos,
                   'pos_from_midline': np.nan,
                   'motor_correlation': np.nan,
                   'left_correlation': np.nan,
                   'right_correlation': np.nan})

# Get pos from midline


def get_pos_from_midline(cell, midline, direction, df, pixel_size):

    if direction == 1:
        pixel_distance = midline - int(df.loc[df.cell == cell, 'x_pos'])
    elif direction == 0:
        pixel_distance = midline - int(df.loc[df.cell == cell, 'y_pos'])

    real_distance = pixel_distance * pixel_size

    return real_distance


pixel_size = float(summary.loc[fish_mask & plane_mask, 'pixel_size'])
direction = int(summary.loc[fish_mask & plane_mask, 'direction'])
df['pos_from_midline'] = pd.Series(cells).apply(get_pos_from_midline, args=(side_lim, direction, df, pixel_size))


# Build regressors

time_indices_bh = np.arange(len(tail_angle))/fps_beh
time_indices_ci = np.arange(dff_f_avg.shape[1])/fps_ci
old_fps = str(round(1/fps_beh, 7))+'S'
new_fps = str(round(1/fps_ci, 7))+'S'
tau = float(summary.loc[fish_mask & plane_mask, 'tau'])
motor_regressor = create_regressor(tail_angle, old_fps, new_fps, time_indices_bh, time_indices_ci,
                                   tau, fps_ci)
plt.savefig(output_path + '/fig/motor_regressor.svg')

df_bout = pd.read_pickle(output_path+'/dataset/df_bout')
tail_angle_left = tail_angle.copy()
tail_angle_left[:] = 0
tail_angle_right = tail_angle_left.copy()
tail_angle_forward = tail_angle_left.copy()

for bout in df_bout.index:
    start, end = df_bout['BoutStart_summed'][bout], df_bout['BoutEnd_summed'][bout]
    if df_bout['abs_Max_Bend_Amp'][bout] < 20:
        tail_angle_forward[start:end] = tail_angle[start:end]
    elif df_bout['Max_Bend_Amp'][bout] <= -20:
        tail_angle_right[start:end] = tail_angle[start:end]
        if any(tail_angle[start:end] > 40):
            tail_angle_left[start:end] = tail_angle[start:end]
    else:
        tail_angle_left[start:end] = tail_angle[start:end]
        if any(tail_angle[start:end] < -40):
            tail_angle_right[start:end] = tail_angle[start:end]

left_regressor = create_regressor(tail_angle_left, old_fps, new_fps, time_indices_bh, time_indices_ci,
                                   tau, fps_ci)
plt.savefig(output_path + '/fig/left_regressor.svg')
right_regressor = create_regressor(tail_angle_right, old_fps, new_fps, time_indices_bh, time_indices_ci,
                                   tau, fps_ci)
plt.savefig(output_path + '/fig/right_regressor.svg')

forward_regressor = create_regressor(tail_angle_forward, old_fps, new_fps, time_indices_bh, time_indices_ci,
                                   tau, fps_ci)
plt.savefig(output_path + '/fig/forward_regressor.svg')

df['motor_correlation'] = pd.Series(cells).apply(pearson_coef, args=(motor_regressor, dff_f_avg_inter))
df['left_correlation'] = pd.Series(cells).apply(pearson_coef, args=(left_regressor, dff_f_avg_inter))
df['right_correlation'] = pd.Series(cells).apply(pearson_coef, args=(right_regressor, dff_f_avg_inter))
df['forward_correlation'] = pd.Series(cells).apply(pearson_coef, args=(forward_regressor, dff_f_avg_inter))

plt.figure()
sns.histplot(df.motor_correlation)

high_corr = df.loc[df.motor_correlation > 0.6, 'cell']
fig, ax = plt.subplots(1, 2)
ax[0].imshow(ops['meanImg'], cmap='Greys', vmax=100)

for i, cell in enumerate(high_corr):
    ax[0].plot(stat[cell]['med'][1], stat[cell]['med'][0], 'o', label=cell)
    ax[1].plot(time_indices_ci, dff_f_avg[cell, :] - i * 50, label=cell)
ax[1].plot(time_trace_bh, tail_angle - 50 * (i + 2))
ax[1].plot(time_trace_f, motor_regressor*10 - 50 * (i + 1))
ax[0].legend()
ax[1].legend()


def build_heatmap_corr(ops, param, cells, stat):
    output = np.zeros(ops['meanImg'].shape)
    output[:] = np.nan
    for cell in cells:
        xpix, ypix = stat[cell]['ypix'], stat[cell]['xpix']
        output[xpix, ypix] = df.loc[df.cell == cell, param]
    return output


fig, ax = plt.subplots(4,1, figsize=(8,12))
vmin = np.nanmin(df[['motor_correlation', 'forward_correlation', 'right_correlation', 'left_correlation']])
vmax = np.nanmax(df[['motor_correlation', 'forward_correlation', 'right_correlation', 'left_correlation']])

for i, corr in enumerate(['motor_correlation', 'forward_correlation', 'right_correlation', 'left_correlation']):
    ax[i].imshow(ops['meanImg'], cmap='Greys', vmax=80)
    corr_map = build_heatmap_corr(ops, corr, set(df.cell), stat)
    pos = ax[i].imshow(corr_map, vmin=-1, vmax=1, cmap='seismic', alpha=0.7)
    ax[i].set_title(corr)
    fig.colorbar(pos, ax=ax[i])
plt.savefig(output_path+'/fig/heatmap_corr.svg')

fig, ax = plt.subplots(4,1, figsize=(8,12))

for i, corr in enumerate(['motor_correlation', 'forward_correlation', 'right_correlation', 'left_correlation']):
    ax[i].imshow(ops['meanImg'], cmap='Greys', vmax=80)
    sns.scatterplot(data=df, x='y_pos', y='x_pos', hue=corr, ax=ax[i], hue_norm=(-1,1), palette='seismic')
    ax[i].set_title(corr)
plt.savefig(output_path+'/fig/heatmap_dot_corr.svg')

plt.figure()
plt.title('Correlation to left (magenta) or right (cyan) regressor as a function of distance from midline')
plt.plot(df.pos_from_midline, df.right_correlation, 'o',color='cyan')
plt.plot(df.pos_from_midline, df.left_correlation, 'o',color='magenta')
plt.ylim(-1,1)
plt.plot([np.nanmin(df.pos_from_midline), np.nanmax(df.pos_from_midline)], [0,0], '--', color='grey')
plt.ylabel('Correlation to sided behavior')
plt.xlabel('Position from midline [um] L/R')
plt.savefig(output_path+'/fig/corr_side_pos_midline.svg')

plt.figure()
sns.scatterplot(data=df, x='left_correlation', y='right_correlation')

df.to_pickle(output_path + '/dataset/df_correlation')

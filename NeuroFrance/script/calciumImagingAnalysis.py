import os, pickle, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from matplotlib import cm
import sys
import math
import shelve
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from utils.import_data import *
from utils.createBoutClass import create_bout_objects
from NeuroFrance.utils.calcium_traces import *


csv_path = '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/chx10_RSN/summaryData_bulbar_group_only.csv'

fishlabel = '200813_F1'
plane = '190um'

summary = pd.read_csv(csv_path)
fish_mask = summary.fishlabel == fishlabel
plane_mask = summary.plane == plane
suite2p_path = list(summary.loc[fish_mask & plane_mask, 'suite2p_path'])[0]
# ZZ_path = list(summary.loc[fish_mask & plane_mask, 'ZZ_path'])[0]
output_path = list(summary.loc[fish_mask & plane_mask, 'output_path'])[0]
fps = float(summary.loc[fish_mask & plane_mask, 'frameRate'])
fps_beh = float(summary.loc[fish_mask & plane_mask, 'frameRateBeh'])

#  get script name and create logging

# script = os.path.basename(__file__)
script = 'calciumImagingAnalysis.py'
handlers = [logging.FileHandler(output_path + '/logs/' + script + '.log'), logging.StreamHandler()]
logging.basicConfig(handlers=handlers,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Import data

## import suite2p output

F, Fneu, spks, stat, ops, iscell = load_suite2p_outputs('', '/', suite2p_path)
time_trace_f = np.arange(F.shape[1])/fps
del suite2p_path



## import behavior info

try:
    tail_angle = np.load(output_path + '/dataset/tail_angle.npy')
    time_trace_bh = np.arange(len(tail_angle)) / fps_beh
except FileNotFoundError:
    behavior_path = summary.loc[fish_mask & plane_mask, 'behavior_path'].item()
    df_bout = pd.read_pickle(behavior_path + '/df_bout_' + plane)
    df_frame = pd.read_pickle(behavior_path + '/df_frame_' + plane)
    tail_angle = np.array(df_frame.Tail_angle)
    time_trace_bh = np.array(df_frame.Time_index)
    df_bout.to_pickle(output_path + '/dataset/df_bout')
    df_frame.to_pickle(output_path + '/dataset/df_frame')
    np.save(output_path + '/dataset/tail_angle.npy', tail_angle)


# Process fluorescence signal

## correct fluo with neuropile

F_corrected, cells = correct_2p_outputs(F, Fneu, iscell)
logging.info('\n\nExtracted '+str(len(cells)) + ' cells.')

layout_F = (go.Layout(title=go.layout.Title(text='Fluorescence of ROIs over time', x=0),
                        yaxis=go.layout.YAxis(title='F', showgrid=False, zeroline=False),
                        xaxis=go.layout.XAxis(title='Time [s]', showgrid=False, zeroline=False)))

tempo = plot_fluo_bh(F_corrected, time_trace_f, tail_angle, time_trace_bh,
                          cells, plane, layout_F, output_path+'/fig/')

## exclude weird cells

to_exclude = str(input('To exclude: (sep by ,)')).split(',')
list_cells = cells.tolist()

try:
    logging.info('\n\nNumber of initial cells:' + str(len(cells)))
    for cell in to_exclude:
        list_cells.remove(int(cell))
    cells = np.array(list_cells)
    logging.info('Excluding cells {}'.format(to_exclude))
    logging.info('\nNumber of remaining cells after exclusion: {}'.format(str(len(cells))))
except ValueError:
    logging.info('\nNo cells excluded.')

del list_cells

## calculate DF/F

noise = np.zeros(F.shape[0])

try:
    dff, base_lims = calc_dff(F_corrected, cells, fps)
    for cell in cells:
        inf, sup = base_lims
        noise[cell] = np.std(dff[cell, inf:sup])
    baseline_lims = [base_lims] * F.shape[0]
    logging.info('\n\nCalculated baseline between frame {} and {} ({} sec)'.format(inf, sup, (sup-inf)/fps))
except IndexError:
    logging.info('\n\nCould not find proper baseline for all cells. Calc baseline for single cell.')
    dff, baseline_lims, noise = calc_dff_single(F_corrected, cells, fps, Fneu)


layout_dff = (go.Layout(title=go.layout.Title(text='Delta fluo over time', x=0),
                        yaxis=go.layout.YAxis(title='Tail angle [°] / Stim [uA.10-1] / DFF [%]', showgrid=False, zeroline=False),
                        xaxis=go.layout.XAxis(title='Time [s]', showgrid=False, zeroline=False)))

tempo = plot_dff_bh(dff, time_trace_f, tail_angle, time_trace_bh, cells, plane,
                         layout_dff, output_path+'/fig/')

del tempo

## filter df/f

### average window filtering

avg_window = math.ceil(fps/2)+1
if avg_window % 2 == 0:
    avg_window -= 1

dff_f_avg = np.array(pd.DataFrame(dff).rolling(window=avg_window, axis=1).mean())
noise_f_avg = np.array(pd.Series(cells).apply(calc_noise, args=(dff_f_avg, baseline_lims)))
logging.info('\n\nFiltered signal using rolling avg, with window: '+str(avg_window))

### Low pass filtering of signal

cut_off = 0.9
dff_f_lp = np.array(dff).copy()
for cell in cells:
    dff_f_lp[cell,:] = lp_filter(cell, dff, fps, cutoff=cut_off)

noise_f_lp = np.array(pd.Series(cells).apply(calc_noise, args=(dff_f_lp, baseline_lims)))
logging.info('\n\nFiltered signal using low pass filter, with cut-off at: '+str(cut_off))

## correct motion artifact

n_per = 1
bad_frames = np.where(ops['corrXY'] < np.percentile(ops['corrXY'], n_per))[0]
logging.info('\n\nFrames with motion artifact: {} (calc using {}th percentile)'.format(bad_frames, n_per))

dff_c = dff_f_lp.copy()
dff_c_rol_avg = dff_f_avg.copy()
dff_c[:, bad_frames] = np.nan
dff_c_rol_avg[:, bad_frames] = np.nan
logging.info('\nNaN-ed {} frames.'.format(len(bad_frames)))

layout_dff = (go.Layout(title=go.layout.Title(text='Delta fluo over time', x=0),
                        yaxis=go.layout.YAxis(title='Tail angle [°] / Stim [uA.10-1] / DFF [%]',
                                              showgrid=False, zeroline=False),
                        xaxis=go.layout.XAxis(title='Time [s]', showgrid=False, zeroline=False)))

tempo = plot_dff_bh(dff_c, time_trace_f, tail_angle, time_trace_bh, cells, plane,
                         layout_dff, output_path+'/fig/lp_filter_')
tempo = plot_dff_bh(dff_c_rol_avg, time_trace_f, tail_angle, time_trace_bh, cells, plane,
                         layout_dff, output_path+'/fig/ra_filter_')

## Interpolate missing data

dff_f_lp_inter = np.array(pd.DataFrame(dff_c).interpolate(axis=1))
dff_f_avg_inter = np.array(pd.DataFrame(dff_c_rol_avg).interpolate(axis=1))

tempo = plot_dff_bh(dff_f_lp_inter, time_trace_f, tail_angle, time_trace_bh, cells, plane,
                         layout_dff, output_path+'/fig/lp_filter_interp_')
del tempo

logging.info('\nFilled NaN using linear interpolation.')

## Save resampled tail angle

nFrames_2p = dff.shape[1]
fq1 = str(round(1/fps_beh, 7)) + 'S'
fq2 = str(round(1/fps, 7)) + 'S'
tail_angle_s2p_gui(tail_angle, nFrames_2p, fq1, fq2, output_path+'/dataset/')

# Align behavior and calcium signal

# Save struct

save_outputs = str(input('Save outputs ? y/(n)'))

if save_outputs == 'y':

    types_to_keep = (np.ndarray, np.generic, int, float, list, pd.core.frame.DataFrame)

    del save_outputs
    del fish_mask
    del plane_mask
    del summary
    del base_lims
    del script
    del sup
    del inf
    del console
    del butter

    shelve_out = shelve.open(output_path + '/shelve_calciumAnalysis.out', 'n')

    for key in dir():
        if isinstance(globals()[key], types_to_keep):
            try:
                shelve_out[key] = globals()[key]
            except TypeError:
                pass

    shelve_out.close()


from NeuroFrance.utils.groupsCells import get_cells_group
from NeuroFrance.utils.calcium_traces import get_pos_y, get_pos_x


summary = pd.read_csv(csv_path)
fish_mask = summary.fishlabel == fishlabel
plane_mask = summary.plane == plane
# cells group
side_lim = summary.loc[fish_mask & plane_mask, 'midline'].item()
cells_group, side, bulbar_lateral, bulbar_medial, pontine, spinal_cord = get_cells_group(summary, fishlabel, plane,
                                                                                         dff, cells, stat)

cells_x_pos = np.array(pd.Series(cells).apply(get_pos_x, args=(stat,)))
cells_y_pos = np.array(pd.Series(cells).apply(get_pos_y, args=(stat,)))


#  sort bulbar medial cells by rostro_caudal pos


if summary.loc[(summary.fishlabel == fishlabel) & (summary.plane == plane), 'direction'].item() == 0:
    bm_pos = np.array([get_pos_x(cell, stat) for cell in bulbar_medial])
    args_pos = bm_pos.argsort()
else:
    bm_pos = np.array([get_pos_y(cell, stat) for cell in bulbar_lateral])
    args_pos = np.flip(bm_pos.argsort())

#  Plot

fig, ax = plt.subplots(1, 2)
fig.suptitle('Example traces of bulbar medial cells during locomotion\n(induced by MLR stim)')
ax[0].imshow(ops['meanImg'], cmap='Greys', vmax=100)
for i, cell in enumerate(np.array(bulbar_lateral)[args_pos]):
    ax[0].plot(stat[cell]['med'][1], stat[cell]['med'][0], 'o', label=cell)
    ax[1].plot(time_trace_f, dff_c_rol_avg[cell, :] - i * 50, label=cell)
ax[1].plot(time_trace_bh, tail_angle - 50 * (i + 1))
ax[0].legend()
ax[1].legend()
plt.savefig(output_path + '/fig/traces_bulbar_lateral.svg')

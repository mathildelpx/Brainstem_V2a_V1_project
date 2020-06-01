import os
import time
import numpy as np
import pandas as pd
import json
import math
import pickle
import base64
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import plot
from utils import Functions_analysis as ff


# function to return a list of index corresponding to element in a list (lst) filling a condition
def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def plot_dff(time_indices, cells_index, state, layout_DFF):
    data = []
    for i in cells_index:
        if i == cells_index[0]:
            data = [ff.data_single_cell_dff(i, DFF, time_indices)]
        else:
            data.append(ff.data_single_cell_dff(i, DFF, time_indices))
    # shift the plot of behavior so it doesn't overlap the calcium signal
    data.append(go.Scatter(name='All bouts ', x=TA_all[:, 0] * shift,
                           y=TA_all[:, 1] - 30))
    fig = go.Figure(data=data, layout=layout_DFF)
    plot(fig,
         filename=output_path + 'fig/' + fishlabel + '/' + trial + '/DFF_' +
                  state + '_' + trial + '.html')
    return fig


def max_DFF_bout_cell(bout, cell, DFF, window_max_DFF):
    """For a given bout and a given cell, find the maximum of DFF value reached by the cell
    between start of the bout and +x second after end of bout
    Time window x can be set by the user (see initialisation of the config_file)
    """
    start = (df_bouts.BoutStartVideo[bout] / fq) * shift
    ca_indices_frame = ff.find_indices(time_indices, lambda e: start < e < start + window_max_DFF)
    DFF_bout_only = DFF[cell, ca_indices_frame]
    try:
        output = np.nanmax(DFF_bout_only)
    except ValueError:
        output = np.nan
    return output


def max_DFF_cell_bout(cell, bout, DFF, window_max_DFF):
    """For a given bout and a given cell, find the maximum of DFF value reached by the cell
    between start of the bout and +x second after end of bout
    Time window x can be set by the used (see initialisation of the config_file
    """
    start = (df_bouts.BoutStartVideo[bout] / fq) * shift
    ca_indices_frame = ff.find_indices(time_indices, lambda e: start < e < start + window_max_DFF)
    DFF_bout_only = DFF[cell, ca_indices_frame]
    try:
        output = np.nanmax(DFF_bout_only)
    except ValueError:
        output = np.nan
    return output


def signal2noise(cell, bout, DFF, noise):
    """Calculates the signal to noise ratio for each cell during a bout.
    """
    output = max_DFF_bout_cell(bout, cell, DFF) / noise[cell]
    return output


def integral_activity(cell, bout, DFF, window_max_DFF):
    start = (df_bouts.BoutStartVideo[bout] / fq)
    ca_indices_frame = ff.find_indices(time_indices, lambda e: start < e < start + window_max_DFF)
    DFF_bout_only = DFF[cell, ca_indices_frame]
    output = sum(DFF_bout_only)
    return output


def get_ROI_DFF(ROI, bad_frames):
    F_corrected = F[ROI] - 0.7 * Fneu[ROI]
    F_corrected[bad_frames] = np.nan
    F0_inf = config_file['F0_inf'][index]
    F0_sup = config_file['F0_sup'][index]
    F0 = np.mean(F_corrected[int(F0_inf):int(F0_sup)])
    DFF = (F_corrected - F0) / F0
    # Define noise as the standard deviation of the baseline.
    noise[ROI] = np.std(DFF[F0_inf:F0_sup])
    return DFF


def plot_heatmap_bout(bout, ops, stat, dff, cells_index, backgroundPath, window_max, dff_type):
    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    for cell in cells_index:
        ypix = stat[cell]['ypix']
        xpix = stat[cell]['xpix']
        heatmap_max[ypix, xpix] = max_DFF_cell_bout(cell, bout, dff, window_max)

    colorscale = [[0.0, 'rgb(49,54,149)'],
                  [0.1, 'rgb(69,117,180)'],
                  [0.2, 'rgb(116,173,209)'],
                  [0.3, 'rgb(171,217,233)'],
                  [0.4, 'rgb(224,243,248)'],
                  [0.5, 'rgb(253,174,97)'],
                  [0.6, 'rgb(244,109,67)'],
                  [0.8, 'rgb(215,48,39)'],
                  [0.9, 'rgb(165,0,38)'],
                  [1.0, 'rgb(165,0,38)']]

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string
    max = np.nanmax(heatmap_max)
    data = go.Heatmap(z=heatmap_max, zmin=0, zmax=max, colorscale=colorscale)
    layout = go.Layout(title='Max ' + dff_type + ' during bout '+str(bout),
                       xaxis=dict(range=[ops['Lx'], 0], showgrid=False),
                       yaxis=dict(range=[0, ops['Ly']], showgrid=False),
                       images=[dict(source=background,
                                    xref='x', yref='y',
                                    x=0, y=0, sizex=ops['Lx'],
                                    sizey=ops['Ly'], xanchor='right', yanchor='bottom',
                                    sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)
    plot(fig_heatmap,
         filename=output_path + 'fig/' + fishlabel + '/' + trial +
                  '/single_bout_vizu/' + str(bout) + '/heatmap_max_activation_' + dff_type + '.html', auto_open=False)


def plot_heatmap_noise_bout(bout, ops, stat, dff, noise, cells_index, backgroundPath, window_max, dff_type):
    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    for cell in cells_index:
        ypix = stat[cell]['ypix']
        xpix = stat[cell]['xpix']
        heatmap_max[ypix, xpix] = max_DFF_cell_bout(cell, bout, dff, window_max)/noise[cell]

    colorscale = [[0.0, 'rgb(49,54,149)'],
                  [0.1, 'rgb(69,117,180)'],
                  [0.2, 'rgb(116,173,209)'],
                  [0.3, 'rgb(171,217,233)'],
                  [0.4, 'rgb(224,243,248)'],
                  [0.5, 'rgb(253,174,97)'],
                  [0.6, 'rgb(244,109,67)'],
                  [0.8, 'rgb(215,48,39)'],
                  [0.9, 'rgb(165,0,38)'],
                  [1.0, 'rgb(165,0,38)']]

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string
    max = np.nanmax(heatmap_max)
    data = go.Heatmap(z=heatmap_max, zmin=0, zmax=max, colorscale=colorscale)
    layout = go.Layout(title='Max ' + dff_type + '/noise during bout '+str(bout),
                       xaxis=dict(range=[ops['Lx'], 0], showgrid=False),
                       yaxis=dict(range=[0, ops['Ly']], showgrid=False),
                       images=[dict(source=background,
                                    xref='x', yref='y',
                                    x=0, y=0, sizex=ops['Lx'],
                                    sizey=ops['Ly'], xanchor='right', yanchor='bottom',
                                    sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)
    plot(fig_heatmap,
         filename=output_path + 'fig/' + fishlabel + '/' + trial +
                  '/single_bout_vizu/' + str(bout) + '/heatmap_max_activation__' + dff_type + '_noise.html', auto_open=False)


def plot_heatmap_integral_bout(bout, ops, stat, dff, cells_index, backgroundPath, window_max, dff_type):
    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    for cell in cells_index:
        ypix = stat[cell]['ypix']
        xpix = stat[cell]['xpix']
        heatmap_max[ypix, xpix] = integral_activity(cell, bout, dff, window_max)

    colorscale = [[0.0, 'rgb(49,54,149)'],
                  [0.1, 'rgb(69,117,180)'],
                  [0.2, 'rgb(116,173,209)'],
                  [0.3, 'rgb(171,217,233)'],
                  [0.4, 'rgb(224,243,248)'],
                  [0.5, 'rgb(253,174,97)'],
                  [0.6, 'rgb(244,109,67)'],
                  [0.8, 'rgb(215,48,39)'],
                  [0.9, 'rgb(165,0,38)'],
                  [1.0, 'rgb(165,0,38)']]

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string
    max = np.nanmax(heatmap_max)
    data = go.Heatmap(z=heatmap_max, zmin=0, zmax=max, colorscale=colorscale)
    layout = go.Layout(title='Sum ' + dff_type + ' during bout '+str(bout),
                       xaxis=dict(range=[ops['Lx'], 0], showgrid=False),
                       yaxis=dict(range=[0, ops['Ly']], showgrid=False),
                       images=[dict(source=background,
                                    xref='x', yref='y',
                                    x=0, y=0, sizex=ops['Lx'],
                                    sizey=ops['Ly'], xanchor='right', yanchor='bottom',
                                    sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)
    plot(fig_heatmap,
         filename=output_path + 'fig/' + fishlabel + '/' + trial +
                  '/single_bout_vizu/' + str(bout) + '/heatmap_sum_activation__' + dff_type + '.html', auto_open=False)


####################################################################################################

fishlabel = '190104_F2'
trial = '7'
index = 11
shift = 1
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'


############################################################################


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


config_file, output_struct = load_output(fishlabel, trial, output_path)

nTrials, frame_rate_2p, fq = len(config_file['Trial_num']), config_file['Frame_rate_2p'][index], \
                             config_file['Fq_camera'][index]
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
filtered_dff = output_struct['filtered_dff']
filtered_dff = np.array(filtered_dff)
bg_path = '/network/lustre/iss01/wyart/rawdata/2pehaviour/190104_F2/background_7.png'
filtered_noise = output_struct['filtered_noise']

nROI, nFrames = F.shape
# create time points to plot everything according to time and not frame index.
time_indices = [frame * (1 / frame_rate_2p) for frame in range(nFrames)]

for bout in df_bouts.index:
    plot_heatmap_bout(bout, ops, stat, DFF, cells_index, bg_path, 1, 'raw_DFF')
    plot_heatmap_bout(bout, ops, stat, filtered_dff, cells_index, bg_path, 1, 'filtered_DFF')
    plot_heatmap_noise_bout(bout, ops, stat, filtered_dff, filtered_noise, cells_index, bg_path, 1, 'filtered_DFF')
    plot_heatmap_noise_bout(bout, ops, stat, DFF, noise, cells_index, bg_path, 1, 'raw_DFF')
    plot_heatmap_integral_bout(bout, ops, stat, DFF, cells_index, bg_path, 1, 'raw_DFF')
    plot_heatmap_integral_bout(bout, ops, stat, filtered_dff, cells_index, bg_path, 1, 'filtered_DFF')

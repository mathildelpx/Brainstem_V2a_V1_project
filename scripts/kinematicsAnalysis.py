import os
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from utils.get_kinematics import get_frames_time_points, tau_calculation
from utils.import_data import load_output_struct, load_suite2p_outputs, load_config_file

# Initialization: write here file name, path.
fishlabel = '190108_F1'
trial = '8'
file_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/dataset/'
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/Tau_calculation/all_roi/' + fishlabel + '_' + trial + '/'
# if output folder does not exists, create it
if not os.path.exists(output_path):
    os.mkdir(output_path)

# signal cut
begin, end = (106, 139)
reanalyze = True

output_struct, F_corrected, DFF, cells_index, noise, time_indices, signal2noise, TA_all, df_bout = load_output_struct(
    fishlabel, trial, file_path)
config_file, nTrials, frame_rate_2p = load_config_file(fishlabel, trial, index=0, path=file_path)
F, Fneu, spks, stat, ops, iscell = load_suite2p_outputs(fishlabel + '/' + trial,
                                                        '/network/lustre/iss01/wyart/analyses/2pehaviour/suite2p_output/')
fq = frame_rate_2p

# cut and put in percentages
dff_cut = DFF[:, begin:end] * 100

# import existing df kinematics or create it if not exists
try:
    df_kinematics = pd.read_pickle(output_path + 'df_kinematics_all_cells')
except FileNotFoundError:
    df_kinematics = pd.DataFrame(columns=['frame_peak', 'frame_end_of_decay',
                                          'tau', 'a', 'b', 'c', 'pcov'],
                                 index=['cell' + str(cell) for cell in cells_index],
                                 dtype=object)

# Plot dff
plt.close()
plt.figure('DFF')
for cell in cells_index:
    plt.plot(dff_cut[cell])
plt.ginput(timeout=2)
plt.close()

# asks users to click for peak and end of curve, fill dataframe
if reanalyze:
    for cell in cells_index:
        output = get_frames_time_points(cell, dff_cut[cell], fq, output_path)
        df_kinematics.loc['cell' + str(cell), 'frame_peak'] = output[0]
        df_kinematics.loc['cell' + str(cell), 'frame_end_of_decay'] = output[1]
        print('Summary cell' + str(cell))
        print(df_kinematics.loc['cell' + str(cell)])

    for cell in cells_index:
        plt.ion()
        roi = cell
        cell = str(cell)
        tau, popt, pcov = tau_calculation(cell, dff_cut[roi], df_kinematics, fq, output_path)
        # returns nan if encountered nan values in the signal to fit, or if peak and end not defined
        if not np.isnan(tau):
            save_tau = input('expo fit ok ? (yes)/no')  # ask user if the fit is ok
            if save_tau != 'no':  # if it entered something else than no, save it
                df_kinematics.at['cell' + cell, 'tau'] = tau
                df_kinematics.at['cell' + cell, 'a'] = popt[0]
                df_kinematics.at['cell' + cell, 'b'] = popt[1]
                df_kinematics.at['cell' + cell, 'c'] = popt[2]
                df_kinematics.at['cell' + cell, 'pcov'] = pcov

df_kinematics.to_pickle(output_path + 'df_kinematics_all_cells')
df_kinematics.to_csv(output_path + 'df_kinematics_all_cells.csv')

heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
heatmap_max[:] = np.nan
for cell in cells_index:
    ypix = stat[cell]['ypix'][~stat[cell]['overlap']]
    xpix = stat[cell]['xpix'][~stat[cell]['overlap']]
    heatmap_max[ypix, xpix] = df_kinematics.at['cell' + str(cell), 'tau']


def heatmap_generator(ops, max):
    data = go.Heatmap(z=heatmap_max, zmin=0, zmax=max, colorscale=colorscale_tau)
    layout = go.Layout(title='Tau on one event',
                       xaxis=dict(range=[ops['Lx'], 0], showgrid=False),
                       yaxis=dict(range=[0, ops['Ly']], showgrid=False),
                       images=[dict(source=background,
                                    xref='x', yref='y',
                                    x=0, y=0, sizex=ops['Lx'],
                                    sizey=ops['Ly'], xanchor='right', yanchor='bottom',
                                    sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)
    plot(fig_heatmap, filename=output_path + 'heatmap_tau.html', auto_open=False)


backgroundPath = '/network/lustre/iss01/wyart/rawdata/2pehaviour/' + fishlabel + '/' + 'background_' + trial + '.png'
with open(backgroundPath, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
# add the prefix that plotly will want when using the string as source
background = "data:image/png;base64," + encoded_string

colorscale_tau = [[0.0, 'rgb(49,54,149)'],
                  [0.1, 'rgb(69,117,180)'],
                  [0.2, 'rgb(116,173,209)'],
                  [0.3, 'rgb(171,217,233)'],
                  [0.4, 'rgb(224,243,248)'],
                  [0.5, 'rgb(253,174,97)'],
                  [0.6, 'rgb(244,109,67)'],
                  [0.8, 'rgb(215,48,39)'],
                  [0.9, 'rgb(165,0,38)'],
                  [1.0, 'rgb(165,0,38)']]

heatmap_generator(ops, np.nanmax(df_kinematics['tau']))

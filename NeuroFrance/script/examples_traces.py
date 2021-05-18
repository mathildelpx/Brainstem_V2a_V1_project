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

fishlabel = '190514_F3'
plane = '10um'

summary = pd.read_csv(csv_path)
fish_mask = summary.fishlabel == fishlabel
plane_mask = summary.plane == plane
suite2p_path = list(summary.loc[fish_mask & plane_mask, 'suite2p_path'])[0]
# ZZ_path = list(summary.loc[fish_mask & plane_mask, 'ZZ_path'])[0]
output_path = list(summary.loc[fish_mask & plane_mask, 'output_path'])[0]
fps = float(summary.loc[fish_mask & plane_mask, 'frameRate'])
fps_beh = float(summary.loc[fish_mask & plane_mask, 'frameRateBeh'])

with open(output_path + '/struct', 'rb') as f:
    struct = pickle.load(f)

df_bout = pd.read_pickle('/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/dataset/'
                         '200813_F1/df_bout_plane_160um')


fig, ax = plt.subplots(1, 2)
fig.suptitle('Example traces of bulbar medial cells during locomotion\n(induced by MLR stim)')
ax[0].imshow(ops['meanImg'], cmap='Greys', vmax=100)

for i, cell in enumerate([45,256,71,53,27,264,21,46]):
    ax[0].plot(stat[cell]['med'][1], stat[cell]['med'][0], 'o', label=cell)
    ax[1].plot(time_indices_ci, dff_f[cell, :] - i * 50, label=cell)
ax[1].plot(time_trace_bh, tail_angle_real - 50 * (i + 1))
ax[0].legend()
ax[1].legend()


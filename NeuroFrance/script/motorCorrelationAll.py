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

summary = pd.read_csv(csv_path)

dict_corr = {}

for fishlabel in set(summary.fishlabel):
    fish_mask = summary.fishlabel == fishlabel
    for plane in set(summary.loc[fish_mask, 'plane']):
        plane_mask = summary.plane == plane
        try:
            output_path = summary.loc[fish_mask & plane_mask, 'output_path'].item()
            df_correlation = pd.read_pickle(output_path + '/dataset/df_correlation')
            print(fishlabel, plane)
        except (FileNotFoundError, TypeError):
            continue
        dict_corr[fishlabel + '_' + plane] = df_correlation

df_correlation_all = pd.concat(dict_corr, ignore_index=True)

df_to_plot = df_correlation_all[
    (df_correlation_all.fishlabel.isin(['200813_F1', '200930_F1'])) & ~(df_correlation_all.group.isin(['spinal_cord',
                                                                                                       'pontine']))]

fig, ax = plt.subplots()
fig.suptitle('Corrleation to left (blue) or right (red) regressor as a function of pos from midline')
sns.kdeplot(data=df_to_plot, x='pos_from_midline', y='right_correlation', ax=ax, cmap='Reds')
sns.kdeplot(data=df_to_plot, x='pos_from_midline', y='left_correlation', ax=ax, cmap='Blues')
ax.set_xlabel('Position from midline [um] L/R')
plt.savefig(
    '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/chx10_RSN/figures_NeuroFrance/motor_correlation/side_corr_pos_midline.svg')

plt.style.use('seaborn-poster')
df_to_plot.index = np.arange(len(df_to_plot))
df_to_plot['side'] = np.nan
for i in df_to_plot.index:
    if df_to_plot.pos_from_midline[i] < 0:
        side = 'L'
    else:
        side = 'R'
    df_to_plot.side[i] = side

fig, ax = plt.subplots()
sns.scatterplot(data=df_to_plot[df_to_plot.fishlabel == '200813_F1'], y='left_correlation', x='right_correlation',
                ax=ax, hue='side', palette='bwr')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
plt.savefig(
    '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/chx10_RSN/figures_NeuroFrance/motor_correlation/l_vs_r.svg')

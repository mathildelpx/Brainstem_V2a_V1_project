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

summary = pd.read_csv(csv_path)

dict_syllabus_all = {}
dict_syl_roi_all = {}
dict_df = {}

for fishlabel in set(summary.fishlabel):
    fish_mask = summary.fishlabel == fishlabel
    for plane in set(summary.loc[fish_mask, 'plane']):
        plane_mask = summary.plane == plane
        try:
            output_path = summary.loc[fish_mask & plane_mask, 'output_path'].item()
            df_correlation = pd.read_pickle(output_path + '/dataset/df_syllabus_roi')
            print(fishlabel, plane)
        except (FileNotFoundError, TypeError):
            continue
        dict_df[fishlabel + '_' + plane] = df_correlation

df_all = pd.concat(dict_df, ignore_index=True)
df = df_all[(df_all.fishlabel == '190514_F3') & (df_all.plane == '10um')]

fig, ax = plt.subplots(1,3)
for i, group in enumerate(['bulbar_medial', 'bulbar_lateral_right', 'bulbar_lateral_left']):
    sns.scatterplot(data=df[df.roi_group == group], x='max_ta', y='max_dff', ax=ax[i])

fig, ax = plt.subplots(1,3)
for i, group in enumerate(['bulbar_medial', 'bulbar_lateral_right', 'bulbar_lateral_left']):
    sns.scatterplot(data=df[df.roi_group == group], x='max_ta', y='max_dff_norm', ax=ax[i])

fig, ax = plt.subplots(1,3)
for i, group in enumerate(['bulbar_medial', 'bulbar_lateral_right', 'bulbar_lateral_left']):
    sns.scatterplot(data=df[df.roi_group == group], x='max_ta', y='dif_dff_start_end', ax=ax[i])

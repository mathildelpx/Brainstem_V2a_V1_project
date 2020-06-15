import pandas as pd
import numpy as np
from utils.combineDataFrames import combineDataFrames
from utils.plotting import *


filesList = pd.read_csv('/home/mathilde.lapoix/Bureau/testBehaviorSummary.csv')
df_bouts_all = combineDataFrames(filesList, path='/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/')

nFish = 2
nTrials = len(filesList)

# replace bouts that were flagged by the category 'Exc' as excluded
nan_cat = np.where(pd.isnull(df_bouts_all.category))[0]
df_bouts_all['category'][nan_cat] = 'Exc'
nan_class = np.where(pd.isnull(df_bouts_all.classification))[0]
df_bouts_all['classification'][nan_class] = 4

kinematics_hist_global(df_bouts_all, nFish, nTrials)
kinematics_violin_per_fish2(df_bouts_all)
plot_violin_kinematics_cat(df_bouts_all, nTrials, nFish)
kinematics_strip(df_bouts_all)

df_bouts_filtered = df_bouts_all.drop(['Name', 'Manual_type', 'BoutStartVideo', 'BoutEndVideo', 'BoutStart_summed',
                                       'BoutEnd_summed', 'Keep', 'Fishlabel', 'iTBF', 'Second_Bend_Amp',
                                       'median_iTBF', 'category', 'mean_tail_angle',
                                       'Tail_angle_sum', 'Side_biais'], axis=1)
sns.pairplot(df_bouts_filtered, hue='classification', size=3)


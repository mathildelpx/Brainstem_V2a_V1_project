import pandas as pd
import numpy as np
from utils.combineDataFrames import combineDataFrames
from utils.plotting import *


filesList = pd.read_csv('/home/mathilde.lapoix/Bureau/testBehaviorSummary.csv')
df_bouts_all = combineDataFrames(filesList, path='/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/')

nFish = 9
nTrials = len(filesList)

# replace bouts that were flagged by the category 'Exc' as excluded
nan_cat = np.where(pd.isnull(df_bouts_all.category))[0]
df_bouts_all['category'][nan_cat] = 'Exc'
nan_class = np.where(pd.isnull(df_bouts_all.classification))[0]
df_bouts_all['classification'][nan_class] = 4

df_summary = pd.DataFrame({'trial_id': filesList['trial_id'],
                           'fishlabel': filesList['fishlabel'],
                           'nBouts': [0]*len(filesList),
                           'duration': [0]*len(filesList),
                           'boutRate': [0]*len(filesList)})
for i, trial_id in enumerate(df_summary['trial_id']):
    df_summary.at[i, 'nBouts'] = len(df_bouts_all[df_bouts_all['Name'] == trial_id])
    df_summary.at[i, 'duration'] = filesList[filesList['trial_id'] == trial_id]['nFrames'] / \
                                   filesList[filesList['trial_id'] == trial_id]['fq']
df_summary['boutRate'] = df_summary['nBouts']/df_summary['duration']

fishList = ['0']
for fish in list(df_summary['fishlabel']):
    print(fish)
    if str(fish) in list(fishList):
        pass
    else:
       fishList = fishList + [str(fish)]


def getCatProp(fishlabel, category):
    df_fish = df_bouts_all[df_bouts_all['Fishlabel'] == fishlabel]
    NBoutsCat = len(np.where(df_fish['category'] == category)[0])
    output = NBoutsCat/len(df_fish)*100
    return output


df_summary_fish = pd.DataFrame({'propF': pd.Series(fishList).apply(getCatProp, args=('F',)),
                                'propR': pd.Series(fishList).apply(getCatProp, args=('R',)),
                                'propL': pd.Series(fishList).apply(getCatProp, args=('L',)),
                                'propO': pd.Series(fishList).apply(getCatProp, args=('0',)),
                                'propExc': pd.Series(fishList).apply(getCatProp, args=('Exc',))},
                               )

df_summary_fish.plot(kind='bar', stacked='true')
plt.legend(bbox_to_anchor=(1.1, 1.05))


def exampleFishTrials(trial_id, category):
    df_trial = df_bouts_all[df_bouts_all['Name'] == trial_id]
    nBoutsCat = len(np.where(df_trial['category'] == category)[0])
    output = nBoutsCat/len(df_trial)*100
    return output


def trialsDifExample(fishlabel):
    exampleFish = df_summary[df_summary['fishlabel'] == fishlabel]
    exampleTrialsList = exampleFish['trial_id']
    exampleFish['propF'] = pd.Series(exampleTrialsList).apply(exampleFishTrials, args=('F',))
    exampleFish['propL'] = pd.Series(exampleTrialsList).apply(exampleFishTrials, args=('L',))
    exampleFish['propR'] = pd.Series(exampleTrialsList).apply(exampleFishTrials, args=('R',))
    exampleFish['propO'] = pd.Series(exampleTrialsList).apply(exampleFishTrials, args=('O',))
    exampleFish['propExc'] = pd.Series(exampleTrialsList).apply(exampleFishTrials, args=('Exc',))
    exampleFish = exampleFish.set_index('trial_id')

    exampleFish[['propF', 'propL', 'propR', 'propO', 'propExc']].plot(kind='bar', stacked=True)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.title('Difference of categories between trials for fish ' + fishlabel)
    plt.xlabel('Trial ID')
    plt.ylabel('Proportion of each cat')
    plt.tight_layout()
    return exampleFish


sns.stripplot(x='fishlabel', y='nBouts', data=df_summary)
sns.stripplot(x='fishlabel', y='boutRate', data=df_summary, jitter=True)
plt.ylabel('Bout Rate [bouts/min]')
plt.title('Bout Rate (number of bouts/min) in the different trials of different fish')

kinematics_hist_global(df_bouts_all, nFish, nTrials)
kinematics_violin_per_fish2(df_bouts_all, nTrials, nFish)
plot_violin_kinematics_cat2(df_bouts_all, nTrials, nFish)
kinematics_strip(df_bouts_all)

df_bouts_filtered = df_bouts_all.drop(['Name', 'Manual_type', 'BoutStartVideo', 'BoutEndVideo', 'BoutStart_summed',
                                       'BoutEnd_summed', 'Keep', 'Fishlabel', 'iTBF', 'Second_Bend_Amp',
                                       'median_iTBF', 'category', 'mean_tail_angle',
                                       'Tail_angle_sum', 'Side_biais'], axis=1)
sns.pairplot(df_bouts_filtered, hue='classification', size=3)



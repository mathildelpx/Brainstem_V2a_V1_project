from utils.import_data import *


def combineDataFrames(filesList, path):
    trials_list = list(filesList['trial_id'])
    for i, trial_id in enumerate(trials_list):
        print(trial_id)
        naming_format = int(filesList[filesList['trial_id'] == trial_id]['naming_format'])
        if naming_format in [1, 3]:
            depth = trial_id.split("_")[-1]
            depth = depth.split(".")[0]
            if naming_format == 3:
                trial = trial_id.split("_")[-2]
            else:
                trial = depth[0:2]
            if len(depth.split("-")) > 1:
                trial = trial + '-1'
        if naming_format == 2:
            trial = trial_id.split("_")[-1]
            trial = trial.split(".")[0]
        fishlabel = trial_id.split('_')[0] + '_' + trial_id.split('_')[1]
        df_frames, df_bouts = load_behavior_dataframe(fishlabel, trial, path)
        df_bouts['Fishlabel'] = [fishlabel] * len(df_bouts)
        if i == 0:
            df_bouts_all = df_bouts
        else:
            df_bouts_all = df_bouts_all.append(df_bouts, ignore_index=True, sort=None)
    return df_bouts_all


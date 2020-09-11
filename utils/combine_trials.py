import pandas as pd
import numpy as np
from os import path


def combineTrials(depth, summary_fish, fishlabel, dir):
    """ Now loads analyzed df bouts and not the raw one."""
    summary_depth = summary_fish[summary_fish['Depth'] == depth]
    for i, trial in enumerate(summary_depth['Trial']):
        try:
            df_frame_trial = pd.read_pickle(dir + 'dataset/' + fishlabel + '/' + fishlabel + '_raw_frame_dataset_' +
                                            depth + '-' + str(int(trial)))
            df_bout_trial = pd.read_pickle(dir + 'dataset/' + fishlabel + '/' + depth + '/df_bout_' + str(int(trial)))
            if i == 0:
                df_frame_plane = df_frame_trial.copy()
                df_bout_plane = df_bout_trial.copy()
            else:
                df_frame_plane = df_frame_plane.append(df_frame_trial, ignore_index=True, sort=None)
                df_bout_plane = df_bout_plane.append(df_bout_trial, ignore_index=True, sort=None)
        except FileNotFoundError:
            continue
    df_frame_plane.to_pickle(dir + 'dataset/' + fishlabel + '/df_frame_plane_' + depth)
    df_bout_plane.to_pickle(dir + 'dataset/' + fishlabel + '/df_bout_plane_' + depth)
    print('Succesfull combination for trials ', list(summary_depth['Trial']), 'for plane ', depth, '!\n')


def createTailAngleArrayPlane(depth, summary_fish, fishlabel, dir, old_freq="0.003333S", new_freq="0.228S"):
    summary_depth = summary_fish[summary_fish['Depth'] == depth]
    if path.exists(dir + '/np_array/' + fishlabel + '/' + depth + '/overall_ta.npy'):
        print('Overall TA array already exists for plane', depth)
    else:
        for i, trial in enumerate(summary_depth['Trial']):
            try:
                df_frame_trial = pd.read_pickle(dir + 'dataset/' + fishlabel + '/' + fishlabel + '_raw_frame_dataset_' +
                                                depth + '-' + str(int(trial)))
                TA_array = np.array(df_frame_trial['Tail_angle'])
                ta = pd.DataFrame(TA_array,
                                  index=pd.date_range(start="00:00:00",
                                                      periods=len(df_frame_trial['Tail_angle']),
                                                      freq=old_freq))
            except FileNotFoundError:
                print('No ZZ output/ DataFrames found for trial ' + depth + '-' + str(int(trial)))
                nFramesBehavior = int(input('Number of Behavior Frames supposed for this trial ?'))
                TA_array = np.array(np.repeat(0, nFramesBehavior))
                ta = pd.DataFrame(TA_array,
                                  index=pd.date_range(start="00:00:00",
                                                      periods=nFramesBehavior,
                                                      freq=old_freq))

            ta_resampled = ta.resample(new_freq).sum()

            nFrames = int(input('Number of 2P frames for trial ' + depth + '-' + str(int(trial))))

            if len(ta_resampled < nFrames):
                to_fill = int(nFrames - len(ta_resampled))
                ta_resampled = ta_resampled.append(pd.DataFrame(np.repeat(0, to_fill)), ignore_index=True)
            elif len(ta_resampled > nFrames):
                ta_resampled = ta_resampled[0:nFrames]
            if i == 0:
                dict_ta = {}
            dict_ta[str(int(trial))] = ta_resampled
        print(dict_ta.keys())
        try:
            overall_ta = dict_ta[str(int(summary_depth['Trial'].iloc[0]))].append(dict_ta[str(int(summary_depth['Trial'].iloc[1]))],
                                                                        ignore_index=True)
            overall_ta = overall_ta.append(dict_ta[str(int(summary_depth['Trial'].iloc[2]))], ignore_index=True)
            overall_ta = overall_ta.append(dict_ta[str(int(summary_depth['Trial'].iloc[3]))], ignore_index=True)
        except KeyError:
            pass
        try:
            np.save(dir + 'np_array/' + fishlabel + '/' + depth + '/overall_ta.npy', np.array(overall_ta))
            print('Succesfull creation of combined tail angle array for trials ', summary_depth['Trial'], 'for plane ',
                  depth, '!\n')
        except UnboundLocalError:
            pass


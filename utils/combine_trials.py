import pandas as pd
import numpy as np
from os import path


def combineTrials(depth, summary_fish, fishlabel, output_path, exp):
    """ Now loads analyzed df bouts and not the raw one."""
    # Get the number of trials for this fish loading the summary csv file
    summary_depth = summary_fish[summary_fish['Depth'] == depth]

    # For every trial at this depth, load the DataFrames (bout and frame)
    for i, trial in enumerate(summary_depth['Trial']):
        print('Trial', trial, '\n')

        try:
            summary_trial = summary_depth[summary_depth['Trial'] == trial]
            df_frame_trial = pd.read_pickle(output_path + 'dataset/' + fishlabel + '/' + fishlabel + '_raw_frame_dataset_' +
                                            depth + '-' + str(int(trial)))
            df_bout_trial = pd.read_pickle(output_path + 'dataset/' + fishlabel + '/' + depth + '/df_bout_' + str(int(trial)))

            # The length of the behavior recording and 2P recording are never the same,
            # but we need to compensate for it to align the 2 recordings together and stack the different recordings

            # If the tail angle given by the df_frame is shorter than what there is in the 2P recording,
            # It's because ZZ did not detect movement and/or because sometimes
            # the 2P has a bit more frames than expected.
            # So I just add the remaining tail angle to fill to be 0.

            behavior_frames_to_have = int((summary_trial['nFrames2P']/exp.fps_2p)*exp.fps_beh)
            print('For this trial, we should have', behavior_frames_to_have, 'frames in the behaviour recording. \n')
            if len(df_frame_trial) < behavior_frames_to_have:
                n_frames_to_add = behavior_frames_to_have - len(df_frame_trial)
                print('Adding', n_frames_to_add, 'frames to behavior DataFrame to match 2P recording. \n')
                to_add = pd.DataFrame(columns=df_frame_trial.columns,
                                      index=range(n_frames_to_add))
                to_add['Tail_angle'] = [0] * n_frames_to_add
                df_frame_trial = df_frame_trial.append(to_add, ignore_index=True, sort=None)
            else:
                to_cut = int(len(df_frame_trial) - behavior_frames_to_have)
                print('Cutting', to_cut, 'frames to behavior DataFrame to match 2P recording. \n')
                df_frame_trial = df_frame_trial[:-to_cut, :]

            #  now we can actually build the DataFrames per plane
            if i == 0:
                df_frame_plane = df_frame_trial.copy()
                df_bout_plane = df_bout_trial.copy()
            else:
                # update the start and end index of each bout by adding the length of each trial made before
                df_bout_trial['BoutStart_summed'] = df_bout_trial['BoutStartVideo'] + len(df_frame_plane)
                df_bout_trial['BoutEnd_summed'] = df_bout_trial['BoutEndVideo'] + len(df_frame_plane)
                df_frame_plane = df_frame_plane.append(df_frame_trial, ignore_index=True, sort=None)
                df_bout_plane = df_bout_plane.append(df_bout_trial, ignore_index=True, sort=None)

        except FileNotFoundError:
            continue
    df_frame_plane.to_pickle(output_path + 'dataset/' + fishlabel + '/df_frame_plane_' + depth)
    df_bout_plane.to_pickle(output_path + 'dataset/' + fishlabel + '/df_bout_plane_' + depth)
    print('Succesfull combination for trials ', list(summary_depth['Trial']), 'for plane ', depth, '!\n')


def createTailAngleArrayPlane(depth, summary_fish, fishlabel, output_path, old_freq="0.003333S", new_freq="0.228S"):
    summary_depth = summary_fish[summary_fish['Depth'] == depth]
    if path.exists(output_path + '/np_array/' + fishlabel + '/' + depth + '/overall_ta.npy'):
        print('Overall TA array already exists for plane', depth)
    else:
        for i, trial in enumerate(summary_depth['Trial']):
            try:
                df_frame_trial = pd.read_pickle(output_path + 'dataset/' + fishlabel + '/' + fishlabel + '_raw_frame_dataset_' +
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
            overall_ta = dict_ta[str(int(summary_depth['Trial'].iloc[0]))].append(
                dict_ta[str(int(summary_depth['Trial'].iloc[1]))],
                ignore_index=True)
            overall_ta = overall_ta.append(dict_ta[str(int(summary_depth['Trial'].iloc[2]))], ignore_index=True)
            overall_ta = overall_ta.append(dict_ta[str(int(summary_depth['Trial'].iloc[3]))], ignore_index=True)
        except KeyError:
            pass
        try:
            np.save(output_path + 'np_array/' + fishlabel + '/' + depth + '/overall_ta.npy', np.array(overall_ta))
            print('Succesfull creation of combined tail angle array for trials ', summary_depth['Trial'], 'for plane ',
                  depth, '!\n')
        except UnboundLocalError:
            pass

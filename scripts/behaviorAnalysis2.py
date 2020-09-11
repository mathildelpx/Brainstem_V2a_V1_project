import pandas as pd
from utils.import_data import *
from utils.load_classified_data import *
from utils.functions_behavior import *
from utils.plotting import *
from tools.list_tools import *


pd.options.mode.chained_assignment = None

fishlabel = '200813_F1'
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'
classification_path = '/home/mathilde.lapoix/Bureau/ZZBehaviorAnalysis/results/megacluster_spont_OMR_N2/'
summary_file_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/summaryDataFinal.csv'

threshold_turns = 20
threshold_struggle = 85


try:
    summary_file = pd.read_csv(summary_file_path,
                               header=0)
except FileNotFoundError:
    print('The path to the summary file is not valid.')
    quit()

summary_fish = summary_file[summary_file['Fishlabel'] == fishlabel]

# LOAD CONFIG FILES, STRUCTS

depth = str(input('Depth ?'))

summary_plane = summary_fish[summary_fish['Depth'] == depth]
trials = summary_plane['Trial']

experiment = load_experiment(output_path, fishlabel)
experiment = experiment[0]

fps_beh = experiment.fps_beh
fps_2p = experiment.fps_2p

for i in list(summary_plane['Trial']):
    plt.close()

    trial = int(i)
    trial_id = fishlabel + '_' + depth + '-' + str(trial)

    print('Trial', trial)
    print('Trial_ID', trial_id)

    df_frame_trial = pd.read_pickle(output_path + 'dataset/' + fishlabel + '/' + fishlabel + '_raw_frame_dataset_' +
                                    depth + '-' + str(trial))
    df_bout_trial = pd.read_pickle(output_path + 'dataset/' + fishlabel + '/' + fishlabel + '_raw_bout_dataset_' +
                                   depth + '-' + str(trial))

    nBouts = len(df_bout_trial)

    # Skip this iteration if no bout was detected.
    if nBouts == 0:
        print('No Bout detected for this trial.')
        continue

    df_bout = df_bout_trial.copy()
    df_bout['Keep'] = 1

    print("depth", depth)
    print('frame rate behavior:', fps_beh)
    print('frame rate 2P:', fps_2p)
    print('Total number of bouts: ', nBouts)

    ##################################################"

    # LOAD CLASSIFICATION

    classification_df = load_output_dataframe(classification_path + 'output_dataframe.csv', trial_id)

    if nBouts != len(classification_df):
        print('number of bouts not corresponding between '
              'the data already analyzed by the pipeline and the classified data.')
        print('Either ', nBouts - len(classification_df), 'bouts were not classified, or')
        print('check file name, doublons or date of analysis/acquisition.')

    df_bout['classification'] = get_classification(list(df_bout.index), classification_df, missing_class=2)
    df_bout['category'] = pd.Series(df_bout.index).apply(replace_category2,
                                                         args=(df_bout, df_frame_trial,
                                                               threshold_turns, threshold_struggle))

    for i in range(1, 12):
        if i * i >= len(df_bout_trial):
            output = i
            break
        elif len(df_bout_trial) == 1:
            continue
    n_col, n_row = (output, output)

    # plot bouts with class to check it
    colors = ['#1A5E63', '#32C837', '#19B986', '#0DB2AD', '#00AAD4']
    # get max bout duration to know how long the fixed time scale needs to be
    max_bout_duration = df_bout['Bout_Duration'].max()
    plt.figure(35, figsize=[25, 15])
    plt.suptitle(fishlabel+' ' + depth + '-' + str(trial) + '\nTail angle for each bout (color coded by cluster)')
    for i, bout in enumerate(range(nBouts)):
        plt.subplot(n_row, n_col, i + 1)
        classification = df_bout.classification[bout]
        color = colors[int(classification)]
        # plot each bout from start, with a fixed time scale (take the max bout duration to scale every other).
        start = df_bout.BoutStartVideo[bout]
        end = df_bout.BoutEndVideo[bout]
        end_max = start + int(max_bout_duration * fps_beh)
        # to avoid plotting the coming bouts, I fill the tail angle after the end of the bout with 0.
        if end_max != end:
            to_fill = len(range(0, end_max - start)) - len(df_frame_trial.Tail_angle[start:end])
            tail_angle = list(df_frame_trial.Tail_angle[start:end]) + [0] * to_fill
            bend_amp = list(df_frame_trial.Bend_Amplitude[start:end]) + [np.nan] * to_fill
            try:
                plt.plot(df_frame_trial.Time_index[start:end_max],
                         tail_angle, color=color,
                         label=classification)
                plt.plot(df_frame_trial.Time_index[start:end_max] - (1 / fps_beh),
                         bend_amp, 'rx', markersize=1.5)
            except ValueError:
                # last bout, the time index cannot go beyond
                try:
                    plt.plot(range(start, end_max),
                             tail_angle, color=color,
                             label=classification)
                    plt.plot(range(start, end_max),
                             bend_amp, 'rx', markersize=1.5)
                except ValueError:
                    pass
        else:
            plt.plot(df_frame_trial.Time_index[start:end], df_frame_trial.Tail_angle[start:end], color=color,
                     label=classification)
            plt.plot(df_frame_trial.Time_index[start:end] - (1 / fps_beh), df_frame_trial.Bend_Amplitude[start:end],
                     'rx', markersize=1.5)
        plt.ylim(-120, 120)
        plt.title('B ' + str(i))
        if i == 0:
            plt.ylabel('Tail angle [°]')
        if i == ((n_row - 1) * n_col):
            plt.xlabel('Time (s)')
            plt.ylabel('Tail angle [°]')
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path + 'fig/' + fishlabel + '/' + depth + '/Traces_all_bouts_' + str(trial) + '.png',
                transparent=True)
    plt.savefig(output_path + 'fig/' + fishlabel + '/' + depth + '/Traces_all_bouts_' + str(trial) + '.pdf',
                transparent=True)

    cluster_represented = get_unique_elements(df_bout.classification)
    plt.figure(36, figsize=[25, 15])
    plt.suptitle(fishlabel+' ' + depth + '-' + str(trial) + '\nStacked bouts for each cluster)')
    for i, cluster in enumerate(cluster_represented):
        plt.subplot(len(cluster_represented), 1, i + 1)
        color = colors[int(cluster)]
        for bout in list(df_bout[df_bout['classification'] == cluster].index):
            start = df_bout.BoutStartVideo[bout]
            end = df_bout.BoutEndVideo[bout]
            end_max = start + int(max_bout_duration * fps_beh)
            if end_max != end:
                to_fill = len(range(0, end_max - start)) - len(df_frame_trial.Tail_angle[start:end])
                tail_angle = list(df_frame_trial.Tail_angle[start:end]) + [0] * to_fill
                bend_amp = list(df_frame_trial.Bend_Amplitude[start:end]) + [np.nan] * to_fill
                try:
                    plt.plot(range(0, end_max - start),
                             tail_angle, color=color)
                    plt.plot(pd.Series(range(0, end_max - start))-1,
                             bend_amp, 'rx', markersize=1.5)
                except ValueError:
                    pass
            else:
                plt.plot(range(0, end_max - start), df_frame_trial.Tail_angle[start:end], color=color)
                plt.plot(pd.Series(range(0, end_max - start))-1,
                         df_frame_trial.Bend_Amplitude[start:end], 'rx', markersize=1.5)
            plt.ylim(-120, 120)
        plt.title('Cluster ' + str(cluster))
        if i == 0:
            plt.ylabel('Tail angle [°]')
        if i == len(cluster_represented):
            plt.xlabel('Time (s)')
            plt.ylabel('Tail angle [°]')

    plt.tight_layout()
    plt.savefig(output_path + 'fig/' + fishlabel + '/' + depth + '/stacked_bouts_perCluster_' + str(trial) + '.png')
    plt.savefig(output_path + 'fig/' + fishlabel + '/' + depth + '/stacked_bouts_perCluster_' + str(trial) + '.pdf',
                transparent=True)

    categories_represented = get_unique_elements(df_bout.category)
    colors_cat = {'F': '#FFCBDD',
                  'R': '#FB4B4E',
                  'L': '#7C0B2B',
                  'S': '#3E000C',
                  'Exc': '#BE6B84'}
    plt.figure(37, figsize=[25, 15])
    plt.suptitle(fishlabel+' ' + depth + '-' + str(trial) + '\nStacked bouts for each category')
    for i, category in enumerate(categories_represented):
        plt.subplot(len(categories_represented), 1, i + 1)
        color = colors_cat[category]
        for bout in list(df_bout[df_bout['category'] == category].index):
            start = df_bout.BoutStartVideo[bout]
            end = df_bout.BoutEndVideo[bout]
            end_max = start + int(max_bout_duration * fps_beh)
            if end_max != end:
                to_fill = len(range(0, end_max - start)) - len(df_frame_trial.Tail_angle[start:end])
                tail_angle = list(df_frame_trial.Tail_angle[start:end]) + [0] * to_fill
                bend_amp = list(df_frame_trial.Bend_Amplitude[start:end]) + [np.nan] * to_fill
                try:
                    plt.plot(range(0, end_max - start),
                             tail_angle, color=color)
                    plt.plot(pd.Series(range(0, end_max - start))-1,
                             bend_amp, 'rx', markersize=1.5)
                except ValueError:
                    pass
            else:
                plt.plot(range(0, end_max - start), df_frame_trial.Tail_angle[start:end], color=color)
                plt.plot(pd.Series(range(0, end_max - start))-1, df_frame_trial.Bend_Amplitude[start:end], 'rx', markersize=1.5)

        plt.ylim(-120, 120)
        plt.title('category ' + str(category))
        if i == 0:
            plt.ylabel('Tail angle [°]')
        if i == len(categories_represented):
            plt.xlabel('Time (s)')
            plt.ylabel('Tail angle [°]')

    plt.savefig(output_path + 'fig/' + fishlabel + '/' + depth + '/stacked_bouts_perCat_' + str(trial) + '.png')
    plt.savefig(output_path + 'fig/' + fishlabel + '/' + depth + '/stacked_bouts_perCat_' + str(trial) + '.pdf',
                transparent=True)

    # Exclude bouts if needed

    # exclusion = [int(x) for x in input('Bouts to exclude?').split()]
    # if exclusion:
    #     df_bout.loc[exclusion, 'Keep'] = 0
    #     df_bout = df_bout.drop[np.where(df_bout.Keep == 0)[0]]
    #
    # analysis_log['bouts_excluded'] = exclusion

    # Figure to show kinematics parameters per category

    plot_violin_kinematics_class(df_bout, 2, 3, output_path, fishlabel, str(trial), depth)
    plot_violin_kinematics_cat(df_bout, 2, 3, output_path, fishlabel, depth, trial, colors_cat)
    dimensions = [list(df_bout.columns)[2]] + list(df_bout.columns)[7:14] + list(df_bout.columns)[16:18]
    scatter_matrix_fish_hue(df_bout, dimensions, output_path, fishlabel, depth, str(trial))

    # plt.show()
    # save = str(input('Save ? (y)/n
    # if save != n
    df_bout.to_pickle(output_path + 'dataset/' + fishlabel + '/' + depth + '/df_bout_' + str(trial))
    print('Bout dataframe saved in pickle format.')

    plt.close()


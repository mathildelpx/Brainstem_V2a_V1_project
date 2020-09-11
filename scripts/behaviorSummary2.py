import pandas as pd
import numpy as np
from utils.combineDataFrames import combineDataFrames
from utils.plotting import *


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
fps_beh = 300

try:
    summary_file = pd.read_csv(summary_file_path,
                               header=0)
except FileNotFoundError:
    print('The path to the summary file is not valid.')
    quit()

summary_fish = summary_file[summary_file['Fishlabel'] == fishlabel]

for i, depth in enumerate(get_unique_elements(summary_fish['Depth'])):
    with open(output_path + 'dataset/' + fishlabel + '/df_frame_plane_' + depth, 'rb') as f:
        df_frame = pickle.load(f)
    with open(output_path + 'dataset/' + fishlabel + '/df_bout_plane_' + depth, 'rb') as f:
        df_bout = pickle.load(f)
    print('plane', depth)
    print(df_bout.columns)
    if i == 0:
        df_frame_all = df_frame
        df_bout_all = df_bout
    else:
        if depth == '230um':
            pass
        else:
            df_frame_all = df_frame_all.append(df_frame, ignore_index=True, sort=None)
            df_bout_all = df_bout_all.append(df_bout_all, ignore_index=True, sort=None)

df_frame_all.to_pickle(output_path + 'dataset/' + fishlabel + '/df_frame_allBouts')
df_bout_all.to_pickle(output_path + 'dataset/' + fishlabel + '/df_bout_allBouts')

max_bout_duration = df_bout_all['Bout_Duration'].max()
colors = ['#1A5E63', '#32C837', '#19B986', '#0DB2AD', '#00AAD4']

cluster_represented = get_unique_elements(df_bout_all.classification)
plt.figure(36, figsize=[25, 15])
plt.suptitle(fishlabel+' -allBouts\nStacked bouts for each cluster)')
for i, cluster in enumerate(cluster_represented):
    plt.subplot(len(cluster_represented), 1, i + 1)
    color = colors[int(cluster)]
    for bout in list(df_bout_all[df_bout_all['classification'] == cluster].index):
        start = df_bout_all.BoutStartVideo[bout]
        end = df_bout_all.BoutEndVideo[bout]
        end_max = start + int(max_bout_duration * fps_beh)
        if end_max != end:
            to_fill = len(range(0, end_max - start)) - len(df_frame_all.Tail_angle[start:end])
            tail_angle = list(df_frame_all.Tail_angle[start:end]) + [0] * to_fill
            bend_amp = list(df_frame_all.Bend_Amplitude[start:end]) + [np.nan] * to_fill
            try:
                plt.plot(range(0, end_max - start),
                         tail_angle, color=color)
                plt.plot(pd.Series(range(0, end_max - start))-1,
                         bend_amp, 'rx', markersize=1.5)
            except ValueError:
                pass
        else:
            plt.plot(range(0, end_max - start), df_frame_all.Tail_angle[start:end], color=color)
            plt.plot(pd.Series(range(0, end_max - start))-1,
                     df_frame_all.Bend_Amplitude[start:end], 'rx', markersize=1.5)
        plt.ylim(-120, 120)
    plt.title('Cluster ' + str(cluster))
    if i == 0:
        plt.ylabel('Tail angle [°]')
    if i == len(cluster_represented):
        plt.xlabel('Time (s)')
        plt.ylabel('Tail angle [°]')

plt.tight_layout()
plt.savefig(output_path + 'fig/' + fishlabel + '/stacked_bouts_perCluster_allBouts.png')
plt.savefig(output_path + 'fig/' + fishlabel + '/stacked_bouts_perCluster_allBouts.pdf',
            transparent=True)

categories_represented = get_unique_elements(df_bout_all.category)
colors_cat = {'F': '#FFCBDD',
              'R': '#FB4B4E',
              'L': '#7C0B2B',
              'S': '#3E000C',
              'Exc': '#BE6B84'}
plt.figure(37, figsize=[25, 15])
plt.suptitle(fishlabel+' allBouts\nStacked bouts for each category')
for i, category in enumerate(categories_represented):
    plt.subplot(len(categories_represented), 1, i + 1)
    color = colors_cat[category]
    for bout in list(df_bout_all[df_bout_all['category'] == category].index):
        start = df_bout_all.BoutStartVideo[bout]
        end = df_bout_all.BoutEndVideo[bout]
        end_max = start + int(max_bout_duration * fps_beh)
        if end_max != end:
            to_fill = len(range(0, end_max - start)) - len(df_frame_all.Tail_angle[start:end])
            tail_angle = list(df_frame_all.Tail_angle[start:end]) + [0] * to_fill
            bend_amp = list(df_frame_all.Bend_Amplitude[start:end]) + [np.nan] * to_fill
            try:
                plt.plot(range(0, end_max - start),
                         tail_angle, color=color)
                plt.plot(pd.Series(range(0, end_max - start))-1,
                         bend_amp, 'rx', markersize=1.5)
            except ValueError:
                pass
        else:
            plt.plot(range(0, end_max - start), df_frame_all.Tail_angle[start:end], color=color)
            plt.plot(pd.Series(range(0, end_max - start))-1, df_frame_all.Bend_Amplitude[start:end], 'rx', markersize=1.5)

    plt.ylim(-120, 120)
    plt.title('category ' + str(category))
    if i == 0:
        plt.ylabel('Tail angle [°]')
    if i == len(categories_represented):
        plt.xlabel('Time (s)')
        plt.ylabel('Tail angle [°]')

plt.savefig(output_path + 'fig/' + fishlabel + '/stacked_bouts_perCat_allBouts.png')
plt.savefig(output_path + 'fig/' + fishlabel + '/stacked_bouts_perCat_allBouts.pdf',
            transparent=True)


def plot_violin_kinematics_cat(df_bouts, rows, cols, output_path, fishlabel, palette):
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=("number osc", "bout duration", "abs max bend", "extremes amp", "TA integral", "mean TBF"))
    for category in get_unique_elements(df_bouts.category):

        fig.add_trace(go.Violin(y=df_bouts['Number_Osc'][df_bouts['category'] == category],
                                x=df_bouts['category'][df_bouts['category'] == category],
                                points='all',
                                hoverinfo='text',
                                text=list(df_bouts.index),
                                line_color=palette[category]),
                      row=1, col=1)

        fig.add_trace(go.Violin(y=df_bouts['Bout_Duration'][df_bouts['category'] == category],
                                x=df_bouts['category'][df_bouts['category'] == category],
                                points='all',
                                hoverinfo='text',
                                text=list(df_bouts.index),
                                line_color=palette[category]),
                      row=1, col=2)

        fig.add_trace(go.Violin(y=df_bouts['abs_Max_Bend_Amp'][df_bouts['category'] == category],
                                x=df_bouts['category'][df_bouts['category'] == category],
                                points='all',
                                hoverinfo='text',
                                text=list(df_bouts.index),
                                line_color=palette[category]),
                      row=1, col=3)

        fig.add_trace(go.Violin(y=df_bouts['Max_Bend_Amp'][df_bouts['category'] == category],
                                x=df_bouts['category'][df_bouts['category'] == category],
                                points='all',
                                hoverinfo='text',
                                text=list(df_bouts.index),
                                line_color=palette[category]),
                      row=2, col=1)

        fig.add_trace(go.Violin(y=df_bouts['Min_Bend_Amp'][df_bouts['category'] == category],
                                x=df_bouts['category'][df_bouts['category'] == category],
                                points='all',
                                hoverinfo='text',
                                text=list(df_bouts.index),
                                line_color=palette[category]),
                      row=2, col=1)

        fig.add_trace(go.Violin(y=df_bouts['Integral_TA'][df_bouts['category'] == category],
                                x=df_bouts['category'][df_bouts['category'] == category],
                                points='all',
                                hoverinfo='text',
                                text=list(df_bouts.index),
                                line_color=palette[category]),
                      row=2, col=2)

        fig.add_trace(go.Violin(y=df_bouts['mean_TBF'][df_bouts['category'] == category],
                                x=df_bouts['category'][df_bouts['category'] == category],
                                points='all',
                                hoverinfo='text',
                                text=list(df_bouts.index),
                                line_color=palette[category]),
                      row=2, col=3)

    fig.update_layout(title_text='Kinematics of bouts in each category (based on automatic classification)'
                                 'n='+str(len(df_bouts))+' bouts',
                      showlegend=False)
    fig.update_xaxes(title_text="category", row=1, col=1)

    plot(fig, filename=output_path + 'fig/' + fishlabel + '/kinematics_per_cat_allBouts.html')
    fig.write_image(output_path + 'fig/' + fishlabel + '/kinematics_per_cat_allBouts.png')


def plot_violin_kinematics_class(df_bouts, rows, cols, output_path, fishlabel):
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=("number osc", "bout duration", "abs max bend",
                                        "extremes amp", "TA integral", "mean TBF"))
    fig.add_trace(go.Violin(y=df_bouts['Number_Osc'], x=df_bouts['classification'], points='all', hoverinfo='text',
                            text=list(df_bouts.index)), row=1, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Bout_Duration'], x=df_bouts['classification'], points='all', hoverinfo='text',
                            text=list(df_bouts.index)), row=1, col=2)
    fig.add_trace(go.Violin(y=df_bouts['abs_Max_Bend_Amp'], x=df_bouts['classification'], points='all', hoverinfo='text',
                            text=list(df_bouts.index)), row=1, col=3)
    fig.add_trace(go.Violin(y=df_bouts['Max_Bend_Amp'], x=df_bouts['classification'], points='all', hoverinfo='text',
                            text=list(df_bouts.index)), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Min_Bend_Amp'], x=df_bouts['classification'], points='all', hoverinfo='text',
                            text=list(df_bouts.index)), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Integral_TA'], x=df_bouts['classification'], points='all', hoverinfo='text',
                            text=list(df_bouts.index)), row=2, col=2)
    fig.add_trace(go.Violin(y=df_bouts['mean_TBF'], x=df_bouts['classification'], points='all', hoverinfo='text',
                            text=list(df_bouts.index)), row=2, col=3)
    fig.update_layout(title_text='Kinematics of bouts in each cluster (based on automatic classification), '
                                 'n='+str(len(df_bouts))+' bouts',
                      showlegend=False)
    fig.update_xaxes(title_text="classification", row=1, col=1)
    plot(fig, filename=output_path + 'fig/' + fishlabel + '/kinematics_per_class_allBouts.html')
    fig.write_image(output_path + 'fig/' + fishlabel + '/kinematics_per_class_allBouts.png')


plot_violin_kinematics_class(df_bout, 2, 3, output_path, fishlabel)
plot_violin_kinematics_cat(df_bout, 2, 3, output_path, fishlabel, colors_cat)
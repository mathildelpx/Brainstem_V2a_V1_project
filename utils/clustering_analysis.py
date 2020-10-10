import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import logging


def print_ma(stuff):
    logging.info('Printing this stuff yeah')
    print('This thing wil not be logged:', stuff)


def new_df_cluster(df_long, nbFrames, fishOfInterest):
    tailAngle = pd.DataFrame(df_long, columns=['tailAngles' + str(i) for i in range(1, nbFrames + 1)])
    instFreq = pd.DataFrame(df_long, columns=['instaTBF' + str(i) for i in range(1, nbFrames + 1)])
    instAsym = pd.DataFrame(df_long, columns=['instaAsym' + str(i) for i in range(1, nbFrames + 1)])

    fishlabels = [a.split('_')[0] + '_' + a.split('_')[1] for a in df_long['Trial_ID']]
    maxInstFreq = [np.nanmax(abs(instFreq.loc[i])) for i in list(df_long.index)]
    maxInstAsym = [np.nanmax(abs(instAsym.loc[i])) for i in list(df_long.index)]
    medianInstFreq = [np.nanmedian(abs(instFreq.loc[i])) for i in list(df_long.index)]

    fishInterest = [0] * len(fishlabels)
    for i, j in enumerate(fishlabels):
        if j == fishOfInterest:
            fishInterest[i] = 1

    output = pd.DataFrame({'Fishlabel': fishlabels,
                           'Trial_ID': df_long.Trial_ID,
                           'NumBout': df_long.NumBout,
                           'NumberOfOscillations': df_long.NumberOfOscillations,
                           'meanTBF': df_long.meanTBF,
                           'BoutDuration': df_long.BoutDuration,
                           'maxAmplitude': df_long.maxAmplitude * 57.2958,
                           'maxInstFreq': maxInstFreq,
                           'maxInstAsym': maxInstFreq,
                           'medianInstFreq': medianInstFreq,
                           'classification': df_long.classification,
                           'fishInterest': fishInterest})

    return output


def fig_clustering(new_df, fishOfInterest):
    plt.figure(figsize=(8, 4))
    sns.stripplot(x='Trial_ID', y='classification', hue="Fishlabel", data=new_df)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.figure(figsize=(8, 4))
    sns.stripplot(x='Trial_ID', y='classification', hue="fishInterest", data=new_df)
    plt.title('Focus on fish of interest')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=['Other fish', fishOfInterest])


def fig_params(new_df, path, clusterName):
    plt.figure(figsize=(7, 7))
    sns.stripplot(y='maxAmplitude', x='classification', hue='classification', data=new_df, palette="tab10")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.figure(figsize=(7, 7))
    sns.stripplot(y='maxInstFreq', x='classification', hue='classification', data=new_df, palette="tab10")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim(-0.5,4)

    plt.figure(figsize=(7, 7))
    sns.stripplot(y='meanTBF', x='classification', hue='classification', data=new_df, palette="tab10")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.figure(figsize=(7, 7))
    sns.stripplot(y='NumberOfOscillations', x='classification', hue='classification', data=new_df, palette="tab10")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.figure(figsize=(7, 7))
    sns.scatterplot(y='meanTBF', x='maxAmplitude', hue='classification', data=new_df, palette="tab10")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(path+clusterName+'/scatter_meanTBFvsMaxAmp.png', transparent=True)

    plt.figure(figsize=(7, 7))
    sns.scatterplot(y='maxInstFreq', x='maxAmplitude', hue='classification', data=new_df, palette="tab10")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim(-0.5, 4)
    plt.savefig(path+clusterName+'/scatter_maxInstFreqVSmaxAmp.png', transparent=True)

    plt.figure(figsize=(7, 7))
    sns.scatterplot(y='medianInstFreq', x='maxAmplitude', hue='classification', data=new_df, palette="tab10")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim(-0.05, 0.45)
    plt.savefig(path+clusterName+'/scatter_medianInstFreqVSmaxAmp.png', transparent=True)


def density_plot_per_cat(df_bouts_all, column="NumberOfOscillations"):
    sns.FacetGrid(df_bouts_all, hue="classification", height=6) \
        .map(sns.distplot, column) \
        .add_legend()


def kdensity_plot_per_cat(df_bouts_all, column="NumberOfOscillations"):
    sns.FacetGrid(df_bouts_all, hue="classification", height=6) \
        .map(sns.kdeplot, column) \
        .add_legend()


def plot_violin_kinematics_class(df_bouts, rows, cols, path, clusterName):
    fig = make_subplots(rows=rows,
                        cols=cols,
                        subplot_titles=("number osc", "bout duration", "mean TBF", "max Amplitude", "max Instant Freq",
                                        "Max Instant Asym"))

    fig.add_trace(go.Violin(y=df_bouts['NumberOfOscillations'],
                            x=df_bouts['classification'],
                            points='all',
                            hoverinfo='text',
                            text=list(zip(list(df_bouts.Trial_ID), list(df_bouts.NumBout)))),
                  row=1, col=1)

    fig.add_trace(go.Violin(y=df_bouts['BoutDuration'],
                            x=df_bouts['classification'],
                            points='all'),
                  row=1, col=2)

    fig.add_trace(go.Violin(y=df_bouts['meanTBF'],
                            x=df_bouts['classification'],
                            points='all',
                            hoverinfo='text',
                            text=list(zip(list(df_bouts.Trial_ID), list(df_bouts.NumBout)))),
                  row=1, col=3)

    fig.add_trace(go.Violin(y=df_bouts['maxAmplitude'],
                            x=df_bouts['classification'],
                            points='all',
                            hoverinfo='text',
                            text=list(zip(list(df_bouts.Trial_ID), list(df_bouts.NumBout)))),
                  row=2, col=1)

    fig.add_trace(go.Violin(y=df_bouts['maxInstFreq'],
                            x=df_bouts['classification'],
                            points='all',
                            hoverinfo='text',
                            text=list(zip(list(df_bouts.Trial_ID), list(df_bouts.NumBout)))),
                  row=2, col=2)

    fig.add_trace(go.Violin(y=df_bouts['maxInstAsym'],
                            x=df_bouts['classification'],
                            points='all',
                            hoverinfo='text',
                            text=list(zip(list(df_bouts.Trial_ID), list(df_bouts.NumBout)))),
                  row=2, col=3)

    fig.update_layout(title_text='Kinematics of bouts in each cluster',
                      showlegend=False)
    fig.update_xaxes(title_text="cluster", row=1, col=1)

    plot(fig, filename=path + clusterName + '/kinematics_per_class.html')
    fig.write_image(path + clusterName + '/kinematics_per_class.png')


def kinematics_hist_global(df_bouts_all, path, clusterName):
    nTrials = len(df_bouts_all)
    nFish_set = set(df_bouts_all.Fishlabel)
    nFish_list = list(nFish_set)
    nFish = len(nFish_list)
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=(
                        "number osc", "bout duration", "abs max bend", "extremes amp", "TA integral", "mean TBF"))
    fig.add_trace(go.Histogram(x=df_bouts_all['NumberOfOscillations']), row=1, col=1)
    fig.add_trace(go.Histogram(x=df_bouts_all['BoutDuration']), row=1, col=2)
    fig.add_trace(go.Histogram(x=df_bouts_all['meanTBF']), row=1, col=3)
    fig.add_trace(go.Histogram(x=df_bouts_all['maxAmplitude']), row=2, col=1)
    fig.add_trace(go.Histogram(x=df_bouts_all['maxInstFreq']), row=2, col=2)
    fig.add_trace(go.Histogram(x=df_bouts_all['maxInstAsym']), row=2, col=3)
    fig.update_layout(title_text='Histograms overall analysis, for a total of ' + str(len(df_bouts_all)) + ' bouts, '
                                 + str(nFish) + ' fish, ' + str(nTrials) + ' trials',
                      showlegend=False)
    fig.update_xaxes(title_text="cluster", row=1, col=1)
    plot(fig, filename=path + clusterName + '/kinematics_histogram.html')
    

def pairplot(data, hue):
    sns.pairplot(data, hue="classification", height=3)


def kinematics_strip(df_bouts_all):
    px.strip(df_bouts_all, x='Fishlabel', y='NumberOfOscillations', color='classification')
    px.strip(df_bouts_all, x='Fishlabel', y='BoutDuration', color='classification')
    px.strip(df_bouts_all, x='Fishlabel', y='meanTBF', color='classification')
    px.strip(df_bouts_all, x='Fishlabel', y='maxAmplitude', color='classification')
    px.strip(df_bouts_all, x='Fishlabel', y='maxInstFreq', color='classification')
    px.strip(df_bouts_all, x='Fishlabel', y='maxInstAsym', color='classification')
    
    

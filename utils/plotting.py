import os
import base64
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from utils.get_kinematics import get_frames_time_points, tau_calculation
from utils.import_data import load_output_struct, load_suite2p_outputs, load_config_file
import plotly.express as px


def plot_single_bout_trace(bout, df_frame, df_bout, fq, op):
    """Plot single bout tail angle trace, as a function of time. COlor of the trace will correspond to the type of bout,
     as it was classified (SFS, L, R, O, None). Right is color 0, left is color 1, sfs is color 2, others is color 3.
     Not classified bouts are black."""
    # get max bout duration to know how long the fixed time scale needs to be
    max_bout_duration = df_bout['Bout_Duration'].max()
    colors = ['#FF00FF', '#FF8000', '#FF8C00', '#228B22']
    classification = df_bout.classification[bout]
    if classification == 0:
        cat = df_bout.category[bout]
        # turns on the right side will be first color, here pink
        # if left turns, orange (second color)
        if cat == 'R':
            color = colors[0]
        else:
            color = colors[1]
    else:
        try:
            color = colors[int(classification) + 2]
        except ValueError:
            color = 'black'
    start = df_bout.BoutStartVideo[bout]
    end = start + int(max_bout_duration*fq)
    plt.figure()
    plt.plot(df_frame.Time_index[start:end], df_frame.Tail_angle[start:end], color)
    plt.plot(df_frame.Time_index[start:end], df_frame.Bend_Amplitude[start:end], 'cx', markersize=1.5)
    plt.ylim(-60, 60)
    plt.title('bout number '+str(bout)+' category:'+str(df_bout.category[bout]))
    plt.ylabel('Tail angle')
    plt.xlabel('time in seconds')
    plt.savefig(op+str(bout)+'/tail_angle_trace.png')
    plt.close()


def plot_categories(category, df_bout, df_frame, fq, op):
    """Plot single bout tail angle trace, as a function of time. COlor of the trace will correspond to the type of bout,
     as it was classified (SFS, L, R, O, None). Right is color 0, left is color 1, sfs is color 2, others is color 3.
     Not classified bouts are black."""
    # get max bout duration to know how long the fixed time scale needs to be
    max_bout_duration = df_bout['Bout_Duration'].max()
    colors = ['#FF00FF', '#FF8000', '#FF8C00', '#228B22']
    # select color for each category
    if category == 'R':
        color = colors[0]
    elif category == 'L':
        color = colors[1]
    elif category == 'F':
        color = colors[3]
    else:
        color = 'k'
    # select subset of df_bouts for the category only
    df_cat = df_bout[df_bout['category'] == category]
    # this one does not work when looking for NaN value, need to use this
    if category is np.nan:
        df_cat = df_bout[df_bout.category.isnull()]
    # plot on top of each other the tai angle of each botu in this category
    plt.figure()
    for i, bout in enumerate(list(df_cat.index)):
        start = df_bout.BoutStartVideo[bout]
        end = start + int(max_bout_duration*fq)
        try:
            plt.plot(np.arange(0, max_bout_duration, step=1/fq), df_frame.Tail_angle[start:end], color)
        except ValueError:
            # when last bout , cannot plot the whole trace cause it goes beyond the tail angle that we have
            try:
                plt.plot(np.arange(0, (df_bout.Bout_Duration[bout] + (1 / fq)), step=1/fq),
                         df_frame.Tail_angle[start:end], color)
            except ValueError:
                pass
    plt.ylim(-60, 60)
    plt.ylabel('Tail angle [°]')
    plt.xlabel('time [s]')
    try:
        plt.title('All bout of category: '+category)
        plt.savefig(op + category + '_all_bouts.png')
    except TypeError:
        # when dealing with unclassified bout, category is NaN, so cannot be converted in str
        plt.title('All bout unclassified')
        plt.savefig(op + 'unclassified_all_bouts.png')
    plt.close()


def plot_roi_masks(fishlabel, trial, file_path, output_path, backgroundPath):
    output_struct, F_corrected, DFF, cells_index, noise, time_indices, signal2noise, TA_all, df_bout = load_output_struct(
            fishlabel, trial, file_path)
    F, Fneu, spks, stat, ops, iscell = load_suite2p_outputs(fishlabel + '/' + trial,
                                                            '/network/lustre/iss01/wyart/analyses/2pehaviour/suite2p_output/')

    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    for cell in cells_index:
        ypix = stat[cell]['ypix']
        xpix = stat[cell]['xpix']
        heatmap_max[ypix, xpix] = 1

    colorscale = [[0.0, 'rgba(49,54,149, 0.0)'],
                  [1.0, 'rgba(255,0,255, 0.6)']]

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string

    data = go.Heatmap(z=heatmap_max, zmin=0, zmax=1, colorscale=colorscale)
    layout = go.Layout(title='Tau on one event',
                       xaxis=dict(range=[ops['Lx'], 0], showgrid=False),
                       yaxis=dict(range=[0, ops['Ly']], showgrid=False),
                       images=[dict(source=background,
                                    xref='x', yref='y',
                                    x=0, y=0, sizex=ops['Lx'],
                                    sizey=ops['Ly'], xanchor='right', yanchor='bottom',
                                    sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)
    plot(fig_heatmap, filename=output_path+fishlabel+'/'+trial+'/ROIs_masks.html', auto_open=False)
    return fig_heatmap


def plot_violin_kinematics_cat(df_bouts, rows, cols, output_path, fishlabel, trial):
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=("number osc", "bout duration", "abs max bend", "extremes amp", "TA integral", "mean TBF"))
    fig.add_trace(go.Violin(y=df_bouts['Number_Osc'], x=df_bouts['category'], points='all'), row=1, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Bout_Duration'], x=df_bouts['category'], points='all'), row=1, col=2)
    fig.add_trace(go.Violin(y=df_bouts['abs_Max_Bend_Amp'], x=df_bouts['category'], points='all'), row=1, col=3)
    fig.add_trace(go.Violin(y=df_bouts['Max_Bend_Amp'], x=df_bouts['category'], points='all'), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Min_Bend_Amp'], x=df_bouts['category'], points='all'), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Integral_TA'], x=df_bouts['category'], points='all'), row=2, col=2)
    fig.add_trace(go.Violin(y=df_bouts['mean_TBF'], x=df_bouts['category'], points='all'), row=2, col=3)
    fig.update_layout(title_text='Kinematics of bouts in each category (based on automatic classification)',
                      showlegend=False)
    fig.update_xaxes(title_text="category", row=1, col=1)
    plot(fig, filename=output_path + 'fig/' + fishlabel + '/' + trial + '/kinematics_per_cat.html')
    fig.write_image(output_path + 'fig/' + fishlabel + '/' + trial + '/kinematics_per_cat.png')


def plot_violin_kinematics_class(df_bouts, rows, cols, output_path, fishlabel, trial):
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=("number osc", "bout duration", "abs max bend", "extremes amp", "TA integral", "mean TBF"))
    fig.add_trace(go.Violin(y=df_bouts['Number_Osc'], x=df_bouts['classification'], points='all'), row=1, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Bout_Duration'], x=df_bouts['classification'], points='all'), row=1, col=2)
    fig.add_trace(go.Violin(y=df_bouts['abs_Max_Bend_Amp'], x=df_bouts['classification'], points='all'), row=1, col=3)
    fig.add_trace(go.Violin(y=df_bouts['Max_Bend_Amp'], x=df_bouts['classification'], points='all'), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Min_Bend_Amp'], x=df_bouts['classification'], points='all'), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Integral_TA'], x=df_bouts['classification'], points='all'), row=2, col=2)
    fig.add_trace(go.Violin(y=df_bouts['mean_TBF'], x=df_bouts['classification'], points='all'), row=2, col=3)
    fig.update_layout(title_text='Kinematics of bouts in each classification (based on automatic classification)',
                      showlegend=False)
    fig.update_xaxes(title_text="classification", row=1, col=1)
    plot(fig, filename=output_path + 'fig/' + fishlabel + '/' + trial + '/kinematics_per_class.html')
    fig.write_image(output_path + 'fig/' + fishlabel + '/' + trial + '/kinematics_per_class.png')


def data_single_cell_F(cell_num, F_corrected, time_indices):
    Fluo = go.Scatter(name='Fluorescence '+str(cell_num), x=time_indices,
                            y=F_corrected[cell_num])
    return Fluo


def data_single_cell_dff(cell_num, DFF, time_indices):
    Fluo = go.Scatter(name='DFF '+str(cell_num), x=time_indices,
                            y=DFF[cell_num])
    return Fluo


def plot_fluo(F_corrected, TA_all, cells_index, time_indices, layout, trial, output_path):
    """"Creates figure of all cells fluorescence over time above tail angle variation over time, classified by symetry

    Takes parameter of times_indices.
    Calls function data_single_cell from the project module ff.
    Saves figure in a folder.
    Uses library plotly.

    """
    data = []
    for i in cells_index:
        if i == cells_index[0]:
            data = [data_single_cell_F(i, F_corrected, time_indices)]
        else:
            data.append(data_single_cell_F(i, F_corrected, time_indices))
    data.append(go.Scatter(name='all', x=TA_all[:, 0],
                           y=TA_all[:, 1] - 100))
    fig = go.Figure(data=data, layout=layout)
    plot(fig, output_path + 'Raw_fluo_' + trial + '.html')


def plot_dff(DFF, TA_all, time_indices, cells_index, shift, trial, layout, output_path):
    data = []
    for i in cells_index:
        if i == cells_index[0]:
            data = [data_single_cell_dff(i, DFF, time_indices)]
        else:
            data.append(data_single_cell_dff(i, DFF, time_indices))
    # shift the plot of behavior so it doesn't overlap the calcium signal
    data.append(go.Scatter(name='All bouts ', x=TA_all[:, 0] * shift,
                           y=TA_all[:, 1] - 30))
    fig = go.Figure(data=data, layout=layout)
    plot(fig,
         filename=output_path + 'DFF_' + trial + '.html')
    return fig


def plot_heatmap_max(bout, dff, cells_index):
    return


def kinematics_violin_per_fish2(df_bouts_all, nFish, nTrials):
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=("number osc", "bout duration", "abs max bend", "extremes amp", "TA integral", "mean TBF"))
    fig.add_trace(go.Violin(y=df_bouts_all['Number_Osc'], x=df_bouts_all['Fishlabel'], points='all'), row=1, col=1)
    fig.add_trace(go.Violin(y=df_bouts_all['Bout_Duration'], x=df_bouts_all['Fishlabel'], points='all'), row=1, col=2)
    fig.add_trace(go.Violin(y=df_bouts_all['abs_Max_Bend_Amp'], x=df_bouts_all['Fishlabel'], points='all'), row=1, col=3)
    fig.add_trace(go.Violin(y=df_bouts_all['Max_Bend_Amp'], x=df_bouts_all['Fishlabel'], points='all'), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts_all['Min_Bend_Amp'], x=df_bouts_all['Fishlabel'], points='all'), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts_all['Integral_TA'], x=df_bouts_all['Fishlabel'], points='all'), row=2, col=2)
    fig.add_trace(go.Violin(y=df_bouts_all['mean_TBF'], x=df_bouts_all['Fishlabel'], points='all'), row=2, col=3)
    fig.update_layout(title_text='Kinematics of bouts for each fish analysed, for a total of ' + str(len(df_bouts_all)) + ' bouts, '
                      + str(nFish) + ' fish, ' + str(nTrials) + ' trials',
                      showlegend=False)
    fig.update_xaxes(title_text="category", row=1, col=1)
    plot(fig, filename='test_kinematics_sum.html')


def kinematics_hist_global(df_bouts_all, nFish, nTrials):
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=("number osc", "bout duration", "abs max bend", "extremes amp", "TA integral", "mean TBF"))
    fig.add_trace(go.Histogram(x=df_bouts_all['Number_Osc']), row=1, col=1)
    fig.add_trace(go.Histogram(x=df_bouts_all['Bout_Duration']), row=1, col=2)
    fig.add_trace(go.Histogram(x=df_bouts_all['abs_Max_Bend_Amp']), row=1, col=3)
    fig.add_trace(go.Histogram(x=df_bouts_all['Max_Bend_Amp']), row=2, col=1)
    fig.add_trace(go.Histogram(x=df_bouts_all['Min_Bend_Amp']), row=2, col=1)
    fig.add_trace(go.Histogram(x=df_bouts_all['Integral_TA']), row=2, col=2)
    fig.add_trace(go.Histogram(x=df_bouts_all['mean_TBF']), row=2, col=3)
    fig.update_layout(title_text='Histograms overall analysis, for a total of ' + str(len(df_bouts_all)) + ' bouts, '
                      + str(nFish) + ' fish, ' + str(nTrials) + ' trials',
                      showlegend=False)
    fig.update_xaxes(title_text="category", row=1, col=1)
    plot(fig, filename='test_kinematics_sum2.html')


def scatter_matrix_fish_hue(data, dimensions):
    fig = px.scatter_matrix(data, dimensions, color='Fishlabel')


def pairplot(data, hue):
    sns.pairplot(data, hue="category", size=3)


def kinematics_strip(df_bouts_all):
    px.strip(df_bouts_all, x='Fishlabel', y='Number_Osc', color='category')
    px.strip(df_bouts_all, x='Fishlabel', y='Bout_Duration', color='category')
    px.strip(df_bouts_all, x='Fishlabel', y='abs_Max_Bend_Amp', color='category')
    px.strip(df_bouts_all, x='Fishlabel', y='Max_Bend_Amp', color='category')
    px.strip(df_bouts_all, x='Fishlabel', y='Integral_TA', color='category')
    px.strip(df_bouts_all, x='Fishlabel', y='mean_TBF', color='category')


def density_plot_per_cat(df_bouts_all):
    sns.FacetGrid(df_bouts_all, hue="category", size=6) \
        .map(sns.distplot, "Number_Osc") \
        .add_legend()


def kdensity_plot_per_cat(df_bouts_all):
    sns.FacetGrid(df_bouts_all, hue="category", size=6) \
        .map(sns.kdeplot, "Number_Osc") \
        .add_legend()


def plot_violin_kinematics_cat2(df_bouts, nTrials, nFish):
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=("number osc", "bout duration", "abs max bend", "extremes amp", "TA integral", "mean TBF"))
    fig.add_trace(go.Violin(y=df_bouts['Number_Osc'], x=df_bouts['category'], points='all'), row=1, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Bout_Duration'], x=df_bouts['category'], points='all'), row=1, col=2)
    fig.add_trace(go.Violin(y=df_bouts['abs_Max_Bend_Amp'], x=df_bouts['category'], points='all'), row=1, col=3)
    fig.add_trace(go.Violin(y=df_bouts['Max_Bend_Amp'], x=df_bouts['category'], points='all'), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Min_Bend_Amp'], x=df_bouts['category'], points='all'), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Integral_TA'], x=df_bouts['category'], points='all'), row=2, col=2)
    fig.add_trace(go.Violin(y=df_bouts['mean_TBF'], x=df_bouts['category'], points='all'), row=2, col=3)
    fig.update_layout(title_text='Kinematics of bouts in each category (based on automatic classification) for ' +
                                 str(len(df_bouts)) + ' bouts, in ' + str(nTrials) + ' trials, in ' +
                                 str(nFish) + ' fish',
                      showlegend=False,
                      xaxis={'categoryorder': 'array', 'categoryarray': ['F', 'L', 'R', 'O', 'Exc']},
                      xaxis1={'categoryorder': 'array', 'categoryarray': ['F', 'L', 'R', 'O', 'Exc']},
                      xaxis2={'categoryorder': 'array', 'categoryarray': ['F', 'L', 'R', 'O', 'Exc']},
                      xaxis3={'categoryorder': 'array', 'categoryarray': ['F', 'L', 'R', 'O', 'Exc']},
                      xaxis4={'categoryorder': 'array', 'categoryarray': ['F', 'L', 'R', 'O', 'Exc']},
                      xaxis5={'categoryorder': 'array', 'categoryarray': ['F', 'L', 'R', 'O', 'Exc']},
                      xaxis6={'categoryorder': 'array', 'categoryarray': ['F', 'L', 'R', 'O', 'Exc']},
                      )
    fig.update_xaxes(title_text="category", row=1, col=1)
    plot(fig, filename='violin_kinematics_per_cat.html')


def plot_violin_kinematics_class2(df_bouts, nTrials, nFish):
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=("number osc", "bout duration", "abs max bend", "extremes amp", "TA integral", "mean TBF"))
    fig.add_trace(go.Violin(y=df_bouts['Number_Osc'], x=df_bouts['classification'], points='all'), row=1, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Bout_Duration'], x=df_bouts['classification'], points='all'), row=1, col=2)
    fig.add_trace(go.Violin(y=df_bouts['abs_Max_Bend_Amp'], x=df_bouts['classification'], points='all'), row=1, col=3)
    fig.add_trace(go.Violin(y=df_bouts['Max_Bend_Amp'], x=df_bouts['classification'], points='all'), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Min_Bend_Amp'], x=df_bouts['classification'], points='all'), row=2, col=1)
    fig.add_trace(go.Violin(y=df_bouts['Integral_TA'], x=df_bouts['classification'], points='all'), row=2, col=2)
    fig.add_trace(go.Violin(y=df_bouts['mean_TBF'], x=df_bouts['classification'], points='all'), row=2, col=3)
    fig.update_layout(title_text='Kinematics of bouts in each classification (based on automatic classification) for ' +
                                 str(len(df_bouts)) + ' bouts, in ' + str(nTrials) + ' trials, in ' +
                                 str(nFish) + ' fish',
                      showlegend=False)
    fig.update_xaxes(title_text="classification", row=1, col=1)
    plot(fig, filename='violin_kinematics_per_class.html')


def kdeplot_all_params(df_bouts_all, params):
    i = 0
    for i, param in enumerate(['Number_Osc', 'Bout_Duration', 'abs_Max_Bend_Amp', 'Integral_TA', 'mean_TBF']):
        sns.FacetGrid(df_bouts_all, hue="category", size=6) \
            .map(sns.kdeplot, "Number_Osc") \
            .add_legend()

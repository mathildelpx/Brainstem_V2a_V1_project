import numpy as np
import base64
from matplotlib import cm
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from tools.list_tools import *


def matplot2plotly_cmap(cmap_type, nsamp=200):
    x = (np.linspace(0, 1, nsamp))
    color_mat = (cm.get_cmap(cmap_type)(x) * 255).astype(np.uint8)[:, :3]
    return [[v, "rgb{}".format(tuple(c))] for v, c in zip(x, color_mat)]


def max_DFF_bout_cell(bout, df_bouts, fps_beh, cell, DFF, window_max_DFF, time_indices):
    """For a given bout and a given cell, find the maximum of DFF value reached by the cell
    between start of the bout and +x second after end of bout
    Time window x can be set by the user (see initialisation of the config_file)
    """
    start = (df_bouts.BoutStartVideo[bout] / fps_beh)
    ca_indices_frame = find_indices(time_indices, lambda e: start < e < start + window_max_DFF)
    DFF_bout_only = DFF[cell, ca_indices_frame]
    try:
        output = np.nanmax(DFF_bout_only)
    except ValueError:
        output = np.nan
    return output


def max_DFF_cell_bout(cell, bout, df_bouts, fps_beh, DFF, time_indices, window_max_DFF):
    """For a given bout and a given cell, find the maximum of DFF value reached by the cell
    between start of the bout and +x second after end of bout
    Time window x can be set by the used (see initialisation of the config_file
    """
    start = (df_bouts.BoutStartVideo[bout] / fps_beh)
    ca_indices_frame = find_indices(time_indices, lambda e: start < e < start + window_max_DFF)
    DFF_bout_only = DFF[cell, ca_indices_frame]
    try:
        output = np.nanmax(DFF_bout_only)
    except ValueError:
        output = np.nan
    return output


def signal2noise(cell, bout, DFF, noise):
    """Calculates the signal to noise ratio for each cell during a bout.
    """
    output = max_DFF_bout_cell(bout, cell, DFF) / noise[cell]
    return output


def integral_activity(cell, bout, df_bouts, fps_beh, DFF, time_indices, window_max_DFF):
    start = (df_bouts.BoutStartVideo[bout] / fps_beh)
    ca_indices_frame = find_indices(time_indices, lambda e: start < e < start + window_max_DFF)
    DFF_bout_only = DFF[cell, ca_indices_frame]
    output = sum(DFF_bout_only)
    return output


def get_ROI_DFF(ROI, F, Fneu, noise, bad_frames, analysis_log):
    F_corrected = F[ROI] - 0.7 * Fneu[ROI]
    F_corrected[bad_frames] = np.nan
    F0_inf = analysis_log['F0_inf']
    F0_sup = analysis_log['F0_sup']
    F0 = np.mean(F_corrected[int(F0_inf):int(F0_sup)])
    DFF = (F_corrected - F0) / F0
    # Define noise as the standard deviation of the baseline.
    noise[ROI] = np.std(DFF[F0_inf:F0_sup])
    return DFF


def plot_heatmap_bout(bout, output_struct, window_max, dff_type):

    ops = output_struct['ops']
    cells_index = output_struct['cells_index']
    stat = output_struct['stat']
    df_bouts = output_struct['df_bouts']
    fps_beh = output_struct['fps_beh']
    DFF = output_struct['filtered_dff']
    time_indices = output_struct['time_indices']
    backgroundPath = output_struct['bg_path']

    if type(time_indices) is not np.ndarray:
        time_indices = np.array(time_indices)
    if type(DFF) is not np.ndarray:
        DFF = np.array(DFF)

    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    for cell in cells_index:
        ypix = stat[cell]['ypix']
        xpix = stat[cell]['xpix']
        heatmap_max[ypix, xpix] = max_DFF_cell_bout(cell, bout, df_bouts, fps_beh, DFF, time_indices, window_max)

    colorscale = [[0.0, 'rgb(49,54,149)'],
                  [0.1, 'rgb(69,117,180)'],
                  [0.2, 'rgb(116,173,209)'],
                  [0.3, 'rgb(171,217,233)'],
                  [0.4, 'rgb(224,243,248)'],
                  [0.5, 'rgb(253,174,97)'],
                  [0.6, 'rgb(244,109,67)'],
                  [0.8, 'rgb(215,48,39)'],
                  [0.9, 'rgb(165,0,38)'],
                  [1.0, 'rgb(165,0,38)']]

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string
    max = np.nanmax(heatmap_max)
    data = go.Heatmap(z=heatmap_max, zmin=0, zmax=max, colorscale=colorscale)
    layout = go.Layout(title='Max ' + dff_type + ' during bout ' + str(bout),
                       xaxis=dict(range=[0, ops['Lx']], showgrid=False),
                       yaxis=dict(range=[ops['Ly'], 0], showgrid=False),
                       images=[dict(source=background,
                                    xref='x', yref='y',
                                    x=0, y=0, sizex=ops['Lx'],
                                    sizey=ops['Ly'], xanchor='left', yanchor='top',
                                    sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)
    output_path = output_struct['output_path']
    fishlabel = output_struct['fishlabel']
    depth = output_struct['depth']

    plot(fig_heatmap,
         filename=output_path + 'fig/' + fishlabel + '/' + depth +
                  '/single_bout_vizu/' + str(bout) + '/heatmap_max_activation_' + dff_type + '.html', auto_open=False)


def plot_heatmap_noise_bout(bout, df_bouts, fps_beh, ops, stat, DFF, noise, cells_index, backgroundPath, window_max,
                            dff_type,
                            output_path, fishlabel, depth, time_indices):
    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    for cell in cells_index:
        ypix = stat[cell]['ypix']
        xpix = stat[cell]['xpix']
        heatmap_max[ypix, xpix] = max_DFF_cell_bout(cell, bout, df_bouts, fps_beh, DFF, time_indices, window_max) / \
                                  noise[cell]

    colorscale = [[0.0, 'rgb(49,54,149)'],
                  [0.1, 'rgb(69,117,180)'],
                  [0.2, 'rgb(116,173,209)'],
                  [0.3, 'rgb(171,217,233)'],
                  [0.4, 'rgb(224,243,248)'],
                  [0.5, 'rgb(253,174,97)'],
                  [0.6, 'rgb(244,109,67)'],
                  [0.8, 'rgb(215,48,39)'],
                  [0.9, 'rgb(165,0,38)'],
                  [1.0, 'rgb(165,0,38)']]

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string
    max = np.nanmax(heatmap_max)
    data = go.Heatmap(z=heatmap_max, zmin=0, zmax=max, colorscale=colorscale)
    layout = go.Layout(title='Max ' + dff_type + '/noise during bout ' + str(bout),
                       xaxis=dict(range=[0, ops['Lx']], showgrid=False),
                       yaxis=dict(range=[ops['Ly'], 0], showgrid=False),
                       images=[dict(source=background,
                                    xref='x', yref='y',
                                    x=0, y=0, sizex=ops['Lx'],
                                    sizey=ops['Ly'], xanchor='left', yanchor='top',
                                    sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)
    plot(fig_heatmap,
         filename=output_path + 'fig/' + fishlabel + '/' + depth +
                  '/single_bout_vizu/' + str(bout) + '/heatmap_max_activation__' + dff_type + '_noise.html',
         auto_open=False)


def plot_heatmap_integral_bout(bout, output_struct, window_max, dff_type):
    ops = output_struct['ops']
    cells_index = output_struct['cells_index']
    stat = output_struct['stat']
    df_bouts = output_struct['df_bouts']
    fps_beh = output_struct['fps_beh']
    DFF = output_struct['filtered_dff']
    noise = output_struct['noise']
    time_indices = output_struct['time_indices']
    backgroundPath = output_struct['bg_path']

    if type(time_indices) is not np.ndarray:
        time_indices = np.array(time_indices)
    if type(DFF) is not np.ndarray:
        DFF = np.array(DFF)

    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    for cell in cells_index:
        ypix = stat[cell]['ypix']
        xpix = stat[cell]['xpix']
        heatmap_max[ypix, xpix] = integral_activity(cell, bout, DFF, window_max)

    colorscale = [[0.0, 'rgb(49,54,149)'],
                  [0.1, 'rgb(69,117,180)'],
                  [0.2, 'rgb(116,173,209)'],
                  [0.3, 'rgb(171,217,233)'],
                  [0.4, 'rgb(224,243,248)'],
                  [0.5, 'rgb(253,174,97)'],
                  [0.6, 'rgb(244,109,67)'],
                  [0.8, 'rgb(215,48,39)'],
                  [0.9, 'rgb(165,0,38)'],
                  [1.0, 'rgb(165,0,38)']]

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string
    max = np.nanmax(heatmap_max)
    data = go.Heatmap(z=heatmap_max, zmin=0, zmax=max, colorscale=colorscale)
    layout = go.Layout(title='Sum ' + dff_type + ' during bout ' + str(bout),
                       xaxis=dict(range=[ops['Lx'], 0], showgrid=False),
                       yaxis=dict(range=[0, ops['Ly']], showgrid=False),
                       images=[dict(source=background,
                                    xref='x', yref='y',
                                    x=0, y=0, sizex=ops['Lx'],
                                    sizey=ops['Ly'], xanchor='right', yanchor='bottom',
                                    sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)

    output_path = output_struct['output_path']
    fishlabel = output_struct['fishlabel']
    depth = output_struct['depth']

    plot(fig_heatmap,
         filename=output_path + 'fig/' + fishlabel + '/' + depth +
                  '/single_bout_vizu/' + str(bout) + '/heatmap_sum_activation__' + dff_type + '.html', auto_open=False)


def plot_heatmap_mean_activity_cat(category, output_struct, window_max, dff_type):

    ops = output_struct['ops']
    cells_index = output_struct['cells_index']
    stat = output_struct['stat']
    df_bouts = output_struct['df_bouts']
    fps_beh = output_struct['fps_beh']
    DFF = output_struct['filtered_dff']
    noise = output_struct['noise']
    time_indices = output_struct['time_indices']
    backgroundPath = output_struct['bg_path']

    if type(time_indices) is not np.ndarray:
        time_indices = np.array(time_indices)
    if type(DFF) is not np.ndarray:
        DFF = np.array(DFF)

    bouts = df_bouts[df_bouts['category'] == category].index

    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    heatmap_cat = np.zeros((ops['Ly'], ops['Lx'], len(bouts)))

    for i, bout in enumerate(bouts):
        for cell in cells_index:
            ypix = stat[cell]['ypix']
            xpix = stat[cell]['xpix']
            heatmap_max[ypix, xpix] = max_DFF_cell_bout(cell, bout, df_bouts, fps_beh, DFF, time_indices,
                                                        window_max)
        heatmap_cat[:, :, i] = heatmap_max

    heatmap_cat_mean = np.nanmean(heatmap_cat, axis=2)

    colorscale = matplot2plotly_cmap("viridis")

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string
    max = np.nanmax(heatmap_cat_mean)
    data = go.Heatmap(z=heatmap_cat_mean, zmin=0, zmax=max, colorscale=colorscale)
    layout = go.Layout(title='Mean activation during ' + category + ' bouts (' + dff_type + ', n=' + str(len(bouts)),
                       xaxis=dict(range=[0, ops['Lx']], showgrid=False),
                       yaxis=dict(range=[ops['Ly'], 0], showgrid=False),
                       images=[dict(source=background,
                                    xref='x', yref='y',
                                    x=0, y=0, sizex=ops['Lx'],
                                    sizey=ops['Ly'], xanchor='left', yanchor='top',
                                    sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)

    output_path = output_struct['output_path']
    fishlabel = output_struct['fishlabel']
    depth = output_struct['depth']

    plot(fig_heatmap,
         filename=output_path + 'fig/' + fishlabel + '/' + depth +
                  '/heatmap_mean_activation_' + category + '_' + dff_type + '.html', auto_open=False)


def plot_heatmap_std_activity_cat(category, output_struct, window_max,
                                  dff_type):

    ops = output_struct['ops']
    cells_index = output_struct['cells_index']
    stat = output_struct['stat']
    df_bouts = output_struct['df_bouts']
    fps_beh = output_struct['fps_beh']
    DFF = output_struct['filtered_dff']
    noise = output_struct['noise']
    time_indices = output_struct['time_indices']
    backgroundPath = output_struct['bg_path']

    if type(time_indices) is not np.ndarray:
        time_indices = np.array(time_indices)
    if type(DFF) is not np.ndarray:
        DFF = np.array(DFF)

    bouts = df_bouts[df_bouts['category'] == category].index

    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    heatmap_cat = np.zeros((ops['Ly'], ops['Lx'], len(bouts)))

    for i, bout in enumerate(bouts):
        for cell in cells_index:
            ypix = stat[cell]['ypix']
            xpix = stat[cell]['xpix']
            heatmap_max[ypix, xpix] = max_DFF_cell_bout(cell, bout, df_bouts, fps_beh, DFF, time_indices,
                                                        window_max)
        heatmap_cat[:, :, i] = heatmap_max

    heatmap_cat_std = np.nanstd(heatmap_cat, axis=2)

    colorscale = matplot2plotly_cmap("RdBu_r")

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string
    max = np.nanmax(heatmap_cat_std)
    data = go.Heatmap(z=heatmap_cat_std, zmin=0, zmax=max, colorscale=colorscale)
    layout = go.Layout(
        title='STD of max activation during ' + category + ' bouts (' + dff_type + ', n=' + str(len(bouts)),
        xaxis=dict(range=[0, ops['Lx']], showgrid=False),
        yaxis=dict(range=[ops['Ly'], 0], showgrid=False),
        images=[dict(source=background,
                     xref='x', yref='y',
                     x=0, y=0, sizex=ops['Lx'],
                     sizey=ops['Ly'], xanchor='left', yanchor='top',
                     sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)

    output_path = output_struct['output_path']
    fishlabel = output_struct['fishlabel']
    depth = output_struct['depth']

    plot(fig_heatmap,
         filename=output_path + 'fig/' + fishlabel + '/' + depth +
                  '/heatmap_std_activation_' + category + '_' + dff_type + '.html', auto_open=False)


def plot_heatmap_median_activity_cat(category, output_struct,
                                     window_max, dff_type):

    ops = output_struct['ops']
    cells_index = output_struct['cells_index']
    stat = output_struct['stat']
    df_bouts = output_struct['df_bouts']
    fps_beh = output_struct['fps_beh']
    DFF = output_struct['filtered_dff']
    noise = output_struct['noise']
    time_indices = output_struct['time_indices']
    backgroundPath = output_struct['bg_path']

    if type(time_indices) is not np.ndarray:
        time_indices = np.array(time_indices)
    if type(DFF) is not np.ndarray:
        DFF = np.array(DFF)

    bouts = df_bouts[df_bouts['category'] == category].index

    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    heatmap_cat = np.zeros((ops['Ly'], ops['Lx'], len(bouts)))

    for i, bout in enumerate(bouts):
        for cell in cells_index:
            ypix = stat[cell]['ypix']
            xpix = stat[cell]['xpix']
            heatmap_max[ypix, xpix] = max_DFF_cell_bout(cell, bout, df_bouts, fps_beh, DFF, time_indices,
                                                        window_max)
        heatmap_cat[:, :, i] = heatmap_max

    heatmap_cat_median = np.nanmedian(heatmap_cat, axis=2)

    colorscale = matplot2plotly_cmap("RdBu_r")

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string
    max = np.nanmax(heatmap_cat_median)
    data = go.Heatmap(z=heatmap_cat_median, zmin=0, zmax=max, colorscale=colorscale)
    layout = go.Layout(title='Median activation during ' + category + ' bouts (' + dff_type + ', n=' + str(len(bouts)),
                       xaxis=dict(range=[0, ops['Lx']], showgrid=False),
                       yaxis=dict(range=[ops['Ly'], 0], showgrid=False),
                       images=[dict(source=background,
                                    xref='x', yref='y',
                                    x=0, y=0, sizex=ops['Lx'],
                                    sizey=ops['Ly'], xanchor='left', yanchor='top',
                                    sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)

    output_path = output_struct['output_path']
    fishlabel = output_struct['fishlabel']
    depth = output_struct['depth']

    plot(fig_heatmap,
         filename=output_path + 'fig/' + fishlabel + '/' + depth +
                  '/heatmap_median_activation_' + category + '_' + dff_type + '.html', auto_open=False)


def plot_heatmap_bout_recruitment(bout, df_bouts, fps_beh, ops, stat, DFF, noise, cells_index, backgroundPath,
                                  window_max, dff_type,
                                  output_path, fishlabel, depth, time_indices):
    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    for cell in cells_index:
        ypix = stat[cell]['ypix']
        xpix = stat[cell]['xpix']
        if max_DFF_cell_bout(cell, bout, df_bouts, fps_beh, DFF, time_indices, window_max) < 2 * noise[cell]:
            output = 0
        else:
            output = 1
        heatmap_max[ypix, xpix] = output

    colorscale = [[0.0, 'rgb(0, 48, 73)'],
                  [1.0, 'rgb(245,138,7)']]

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string
    max = np.nanmax(heatmap_max)
    data = go.Heatmap(z=heatmap_max, zmin=0, zmax=max, colorscale=colorscale)
    layout = go.Layout(title='Max ' + dff_type + ' during bout ' + str(bout),
                       xaxis=dict(range=[0, ops['Lx']], showgrid=False),
                       yaxis=dict(range=[ops['Ly'], 0], showgrid=False),
                       images=[dict(source=background,
                                    xref='x', yref='y',
                                    x=0, y=0, sizex=ops['Lx'],
                                    sizey=ops['Ly'], xanchor='left', yanchor='top',
                                    sizing="stretch", opacity=1, layer='below')])
    fig_heatmap = go.Figure(data=[data], layout=layout)
    plot(fig_heatmap,
         filename=output_path + 'fig/' + fishlabel + '/' + depth +
                  '/single_bout_vizu/' + str(bout) + '/heatmap_recruitment_' + dff_type + '.html', auto_open=True)


def heatmap_recruitment(bout, output_struct,
                        window_max, dff_type):

    # TODO: create a list of lists to fill hover info on the heatmap with the cell number

    ops = output_struct['ops']
    cells_index = output_struct['cells_index']
    stat = output_struct['stat']
    df_bouts = output_struct['df_bouts']
    fps_beh = output_struct['fps_beh']
    DFF = output_struct['filtered_dff']
    noise = output_struct['noise']
    time_indices = output_struct['time_indices']
    backgroundPath = output_struct['bg_path']

    if type(time_indices) is not np.ndarray:
        time_indices = np.array(time_indices)
    if type(DFF) is not np.ndarray:
        DFF = np.array(DFF)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}],
               [{"colspan": 2}, None]],
        subplot_titles=("Tail angle", "DFF of recruited cells", "Recruitment map"))

    heatmap_max = np.zeros((ops['Ly'], ops['Lx']))
    heatmap_max[:] = np.nan
    recruited_cells = [-1]
    for cell in cells_index:
        ypix = stat[cell]['ypix']
        xpix = stat[cell]['xpix']
        if max_DFF_cell_bout(cell, bout, df_bouts, fps_beh, DFF, time_indices, window_max) < 2 * noise[cell]:
            output = 0
        else:
            output = 1
            recruited_cells.append(cell)
        heatmap_max[ypix, xpix] = output

    recruited_cells = recruited_cells[1:]

    max = np.nanmax(heatmap_max)

    colorscale = [[0.0, 'rgb(0, 48, 73)'],
                  [1.0, 'rgb(245,138,7)']]

    with open(backgroundPath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # add the prefix that plotly will want when using the string as source
    background = "data:image/png;base64," + encoded_string

    fig.add_trace(go.Heatmap(z=heatmap_max, zmin=0, zmax=max, colorscale=colorscale), row=2, col=1)

    fig.add_trace(scatter_ta(bout, output_struct),
                  row=1, col=1)
    for cell in recruited_cells:
        fig.add_trace(scatter_dff_recruited_cells(cell, bout, output_struct),
                      row=1, col=2)
    fig.update_layout(title='Max ' + dff_type + ' during bout ' + str(bout),
                      xaxis1=dict(title='Time [s]'), yaxis1=dict(title='Tail angle [°]'),
                      xaxis3=dict(range=[0, ops['Lx']], showgrid=False),
                      yaxis3=dict(range=[ops['Ly'], 0], showgrid=False),
                      images=[dict(source=background,
                                   xref='x3', yref='y3',
                                   x=0, y=0, sizex=ops['Lx'],
                                   sizey=ops['Ly'], xanchor='left', yanchor='top',
                                   sizing="stretch", opacity=1, layer='below')])

    output_path = output_struct['output_path']
    fishlabel = output_struct['fishlabel']
    depth = output_struct['depth']

    plot(fig,
         filename=output_path + 'fig/' + fishlabel + '/' + depth +
                  '/single_bout_vizu/' + str(bout) + '/heatmap_recruitment_' + dff_type + '.html', auto_open=True)


def scatter_ta(bout, output_struct):
    fps_beh = output_struct['fps_beh']
    df_frame = output_struct['df_frame']
    df_bout = output_struct['df_bouts']
    colors_cat = output_struct['colors_cat']
    category = df_bout.category[bout]
    color = colors_cat[category]
    start = int(df_bout.BoutStartVideo[bout] - fps_beh)
    end = int(df_bout.BoutEndVideo[bout] + 2 * fps_beh)
    output = go.Scatter(x=df_frame.Time_index[start:end], y=df_frame.Tail_angle[start:end],
                        mode='lines', marker_color=color, text=category)
    return output


def scatter_dff_recruited_cells(cell, bout, output_struct):
    fps_beh = output_struct['fps_beh']
    fps_2p = output_struct['fps_2p']
    df_bout = output_struct['df_bouts']
    DFF = output_struct['filtered_dff']
    time_indices = output_struct['time_indices']
    if type(time_indices) is not np.ndarray:
        time_indices = np.array(time_indices)
    if type(DFF) is not np.ndarray:
        DFF = np.array(DFF)

    start = fps_2p*((df_bout.BoutStartVideo[bout] - fps_beh)/fps_beh)
    end = fps_2p*((df_bout.BoutEndVideo[bout] + 2 * fps_beh)/fps_beh)
    ca_indices_frame = find_indices(time_indices, lambda e: start < e < end)
    DFF_bout_only = DFF[cell, ca_indices_frame]
    output = go.Scatter(x=time_indices[ca_indices_frame],
                        y=DFF_bout_only,
                        mode='lines')
    return output

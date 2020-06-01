import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def get_frames_time_points(cell, dff, fq, path):
    print('For cell', cell)
    plt.close()
    print('Click 2 times to get: peak of calcium transient and end of curve')
    plt.figure('cell '+str(cell))
    plt.plot(dff, 'b')
    plt.show()
    # Get points from user
    points = plt.ginput(2, timeout=0)
    point_name = ['frame peak', 'frame end of decay']
    colors = ['black', 'green']
    # Get frame number of each click
    frames = []
    if points:
        for i, click in enumerate(points):
            x, y = click  # get w and y of the click
            frames.append(int(x))  # store only x value
            plt.plot(x, y, 'x', color=colors[i], markersize=20, label=point_name[i]+': '+str(x))
        # get resulting time points
        plt.legend()
        plt.savefig(path+'Points_kinematics_cell'+str(cell)+'.png')
    else:
        frames = [np.nan, np.nan]
    plt.close()
    return frames


def exp_function(x, a, b, c):
    return a * np.exp(-b*x) + c


# def tau_calculation(event, dff, df_kinematics, fq, path):
#     plt.close()
#     event = str(event)
#     begin = int(df_kinematics.loc['event'+event, 'frame_peak'])
#     end = int(df_kinematics.loc['event'+str(event), 'frame_end_of_decay'])
#     xdata = np.arange(begin,end)  # create array of frames between peak and end of curve
#     xdata_fit = np.arange(0, len(xdata))  # create array of same length but starting from 0
#     ydata = dff[begin:end]  # signal which will be fitted
#     try:
#         popt, pcov = curve_fit(exp_function, xdata_fit, ydata, maxfev=2000)
#         plt.figure('Exponential fit for decay after event'+str(event))
#         plt.plot(dff, color='blue')
#         plt.plot(xdata, ydata, color='orange')
#         plt.plot(xdata, exp_function(xdata_fit, *popt), color='green', label='fitted curve')
#         trick = plt.ginput(1)  # little trick to make the figure appear, show does not work by itself.
#         plt.savefig(path + 'event_' + str(event) + '_expo_fit.png')
#         plt.figure('Zoom expo fit event'+str(event))
#         plt.plot(xdata_fit, ydata, color='orange', label='original signal')
#         plt.plot(xdata_fit, exp_function(xdata_fit, *popt), color='green',
#                  label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#         plt.legend()
#         plt.show()
#         trick = plt.ginput(1)
#     except ValueError:
#         popt = [np.nan] * 3
#         pcov = np.nan
#         print('ValueError, NaN value encountered.Could not fit the curve here.')
#     print('parameters:', popt)
#     print('covariance of parameters:', pcov)
#     tau = (1/popt[1])/fq  # calculate tau the invert of b, divided by frame rate to get it in seconds
#     print('tau:', tau)
#     plt.savefig(path+'event_'+str(event)+'_expo_fit_zoom.png')
#     plt.close()
#     return tau, popt, pcov


def tau_calculation(cell, dff, df_kinematics, fq, path):
    plt.close()
    cell = str(cell)
    try:
        begin = int(df_kinematics.loc['cell'+cell, 'frame_peak'])
        end = int(df_kinematics.loc['cell'+cell, 'frame_end_of_decay'])
        xdata = np.arange(begin,end)  # create array of frames between peak and end of curve
        xdata_fit = np.arange(0, len(xdata))  # create array of same length but starting from 0
        ydata = dff[begin:end]  # signal which will be fitted
        try:
            popt, pcov = curve_fit(exp_function, xdata_fit, ydata, maxfev=2000)
            plt.figure('Exponential fit for decay after cell'+cell)
            plt.plot(dff, color='blue')
            plt.plot(xdata, ydata, color='orange')
            plt.plot(xdata, exp_function(xdata_fit, *popt), color='green', label='fitted curve')
            trick = plt.ginput(1)  # little trick to make the figure appear, show does not work by itself.
            plt.savefig(path + 'cell_' + cell + '_expo_fit.png')
            plt.figure('Zoom expo fit cell'+cell)
            plt.plot(xdata_fit, ydata, color='orange', label='original signal')
            plt.plot(xdata_fit, exp_function(xdata_fit, *popt), color='green',
                     label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
            plt.legend()
            plt.show()
            trick = plt.ginput(1)
        except ValueError:
            popt = [np.nan] * 3
            pcov = np.nan
            print('ValueError, NaN value encountered in the signal. Could not fit the curve here.')
        print('parameters:', popt)
        print('covariance of parameters:', pcov)
        tau = (1/popt[1])/fq  # calculate tau the invert of b, divided by frame rate to get it in seconds
        print('tau:', tau)
        plt.savefig(path+'cell_'+cell+'_expo_fit_zoom.png')
        plt.close()
    except ValueError:
        print('signal is not curve for this roi, not fitting anything.')
        popt = [np.nan] * 3
        pcov = [np.nan]
        tau = np.nan
    return tau, popt, pcov


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

plt.style.use('seaborn-colorblind')


def nandot(X, Y):
    """Sum two arrays, but taking the NaN into account (sum functions does not naturally)"""
    return np.nansum(X * Y)


def calcium_kernel(tau):
    """Builds calcium transient trace, based on a tau value defined by user.

    :param tau: The tau value you want to give to your exponential decay

    :return: exponential function with tau decay
    """
    return lambda x: np.exp(- x / (tau / np.log(2)))


def convolve_regressors(regressor, kernel):
    """ Convolves the regressor with a kernel function
    :param regressor: the regressor, or regressor matrix
    :param kernel:
    :return: the convolved kernel
    """
    return np.convolve(regressor, kernel)[0:len(regressor)]


def create_regressor(tail_angle: np.ndarray, old_fps: str, new_fps: str,
                     time_indices_bh: np.ndarray, time_indices_ci: np.ndarray,
                     tau: float, fps_ci: float):
    """
    Create motor regressor; e.g signal mimicking fluorescence of a cell expressing calcium indicator
    if it was linearlt encoding the behavior of interest.

    :param tail_angle: 1D numpy array, tail angle value for each time step
    :param old_fps: frequency at which tail angle was sampled, in format understood by pandas resample function
    :param new_fps: same but desired frequency to resample it to, format is "fqS" with fq int or float, ex "0.003S"
    :param time_indices_bh: 1D vector of time at each time points in behavior
    :param time_indices_ci: 1D vector of time at each time points in calcium signal
    :param tau: tau decay of calcium kernel
    :param fps_ci: float, frequency of acquisition of the calcium imaging signal
    :return:
    """

    df_ta = pd.DataFrame(tail_angle, index=pd.date_range(start='00:00:00',
                                                         periods=len(tail_angle),
                                                         freq=old_fps))

    ta_resampled = abs(df_ta).resample(new_fps).sum()

    if len(time_indices_ci) != len(ta_resampled):
        to_fill = len(time_indices_ci) - len(ta_resampled)
        ta_resampled = ta_resampled.append(pd.DataFrame(data=[0] * to_fill,
                                                        index=range(to_fill)),
                                           ignore_index=True,
                                           sort=None)

    # TODO: add exception to handle when tail angle is longer than calcium imaging signal.

    ker = calcium_kernel(tau)(np.arange(0, 10, 1 / fps_ci))
    ker_zeros = np.zeros((int(2 * fps_ci)))
    ker2 = np.concatenate((ker_zeros, ker, ker_zeros))

    regressor = zscore(convolve_regressors(np.array(ta_resampled).flatten(), ker))

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    ax[0, 0].plot(time_indices_bh, tail_angle, color='black')
    ax[0, 0].set_title('Raw tail angle')
    ax[0, 0].set_xlabel('Time [s]')
    ax[0, 0].set_ylabel('Tail angle [°]')

    ta_resampled.index = np.arange(len(ta_resampled)) / fps_ci
    ta_resampled.plot(ax=ax[0, 1], color='orange')
    ax[0, 1].set_title('Resampled tail angle to {} Hz'.format(fps_ci))
    ax[0, 1].set_xlabel('Time [s]')
    ax[0, 1].set_ylabel('Tail angle [°]')

    ax[1, 0].plot(np.arange(len(ker2)), ker2, color='silver')
    ax[1, 0].set_title('Calcium kernel to convolve to tail angle (tau:{})'.format(tau))
    ax[1, 0].set_xlabel('Time [s]')
    ax[1, 0].set_ylabel('Arbitrary unit of intensity')

    ax[1, 1].plot(ta_resampled.index, regressor, color='magenta')
    ax[1, 1].set_title('Final regressor')
    ax[1, 1].set_xlabel('Time [s]')
    ax[1, 1].set_ylabel('Arbitrary unit of intensity')
    plt.tight_layout()

    return regressor


# TODO: build syllabus not by looking at it frame by frame in CI, but from looking frame by frames in behavior trace

def build_cat_tail_angle(tail_angle: np.ndarray, bouts, bout_type):
    """

    Build trace with tail angle corresponding to a specific type of syllabus only.

    :param tail_angle: raw tail angle, 1D numpy array
    :param bouts: dict of instances from class Bout
    :param syl_type: desired bout type to keep.
    :return: 1D numpy array with tail angle for syl type only.

    """

    output = np.zeros(tail_angle.shape)
    taken_bouts = []

    for bout in bouts.keys():

        if bouts[bout].cat == bout_type:
            output[bouts[bout].start:bouts[bout].end] = tail_angle[bouts[bout].start:bouts[bout].end]
            taken_bouts.append(bout)

    print('For bout type: {}, \nFound {} bouts({})'.format(bout_type, len(taken_bouts), taken_bouts))

    return output


def mask_value_cell(cell, ops, stat, fct, f_args):
    output = np.zeros(ops['meanImg'].shape)
    output[:] = np.nan

    xpix, ypix = stat[cell]['xpix'], stat[cell]['ypix']
    output[ypix, xpix] = fct(cell, *f_args)

    return output


def build_heatmap_fct(cells, ops, stat, fct, f_args):
    all_heatmaps = pd.Series(cells).apply(mask_value_cell, args=(ops, stat, fct, f_args))

    output = np.sum(all_heatmaps, axis=0)

    return output


def pearson_coef_pixel(pixel, regressor, pixel_traces):
    """ Calculates the pearson coefficient value between two signal.

    :param pixel: numpy array of the pixel intensity in time
    :param regressor: numpy array of whatever signal you want to correlate your pixel intensity with. Here, tail angle trace.

    :return: pearson correlation coefficient between the 2 signals.
    """
    X = pixel_traces[pixel]
    Y = regressor
    # if regressor is shorter, reduce the size of the pixel trace. Could have filled with 0 the remaining, it depends on how sure you are that your regressor is well built.
    # Thing is the pixel value would not really be 0, you would need to take something as the baseline value for this pixel, and add noise.
    if X.shape != Y.shape:
        Y_bis = np.zeros(X.shape)
        Y_bis[0:len(Y)] = Y
        Y = Y_bis
    numerator = nandot(X, Y) - X.shape[0] * np.nanmean(X) * np.nanmean(Y)
    denominator = (X.shape[0] - 1) * np.nanstd(X) * np.nanstd(Y)
    pearson_corr = numerator / denominator
    return pearson_corr


def pearson_coef(cell, regressor, dff):
    """ Calculates the pearson coefficient value between two signal.

    :param pixel: numpy array of the pixel intensity in time
    :param regressor: numpy array of whatever signal you want to correlate your pixel intensity with. Here, tail angle trace.

    :return: pearson correlation coefficient between the 2 signals.
    """
    X = dff[cell,:]
    Y = regressor
    # if regressor is shorter, reduce the size of the pixel trace. Could have filled with 0 the remaining, it depends on how sure you are that your regressor is well built.
    # Thing is the pixel value would not really be 0, you would need to take something as the baseline value for this pixel, and add noise.
    if X.shape != Y.shape:
        Y_bis = np.zeros(X.shape)
        Y_bis[0:len(Y)] = Y
        Y = Y_bis
    numerator = nandot(X, Y) - X.shape[0] * np.nanmean(X) * np.nanmean(Y)
    denominator = (X.shape[0] - 1) * np.nanstd(X) * np.nanstd(Y)
    pearson_corr = numerator / denominator
    return pearson_corr
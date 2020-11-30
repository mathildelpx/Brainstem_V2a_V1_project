import os, datetime
import  numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


class Bout:
    """Class to define each bout as an object.
    Characterised by attributes as different properties in the df bouts DataFrame.
    The method plot allows you to plot the tail angle for this bout."""
    def __init__(self, df_bouts, df_frames, number):
        self.num = number
        self.cat = df_bouts.category[number]
        self.num_osc = df_bouts.Number_Osc[number]
        self.duration = df_bouts.Bout_Duration[number]
        self.max = df_bouts.Max_Bend_Amp[number]
        self.start = df_bouts.BoutStart_summed[number]
        self.end = df_bouts.BoutEnd_summed[number]
        self.ta = df_frames.Tail_angle[self.start:self.end]
        self.bends = df_frames.Bend_Amplitude[self.start:self.end]

    def plot(cls, df_frames, fq=None):
        plt.figure(figsize=(10,7))
        if fq:
            time_indices = np.array(range(cls.start, cls.end))/fq
            plt.plot(time_indices, cls.ta)
            plt.plot(time_indices-(1/fq), cls.bends, 'rx', markersize=1.5)
            plt.xlabel('Time [s]')
        else:
            plt.plot(df_frames.cls.ta)
            plt.plot(df_frames.cls.bends, 'rx', markersize=1.5)
            plt.xlabel('Frame')
        plt.ylim(-90, 90)
        plt.title('Tail angle over time for bout ' + str(cls.num))
        plt.ylabel('Tail angle [°]')


def create_bout_objects(df_bouts, df_frames, output_path):
    """Creates a pandas Series with Bout object for each bout of the df_bouts.
    Saves it in pickle format in the desired direction."""
    bouts = pd.Series([Bout(df_bouts, df_frames, i) for i in df_bouts.index])
    bouts.to_pickle(output_path + 'bouts')
    return bouts


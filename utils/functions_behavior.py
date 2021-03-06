import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def get_classification(bouts, classification_df, missing_class=4):
    """
    Loads the classification attributed to each bout.
    """
    classification = bouts.copy()
    print('nb of clusters:', np.nanmax(classification_df['classification']))
    # missing_class = int(input('How to flag the bouts with missing clustering (excluded bouts)'))
    for bout in bouts:
        df_single_bout = classification_df[classification_df['NumBout'] == float(bout)]
        try:
            output = int(df_single_bout['classification'])
        except TypeError:
            print('Classification missing for bout:', bout)
            output = missing_class
        classification[bout] = output
    return classification


def decisive_bend(bout_num, df_bout, df_frame):
    """
    Way to determine if a bout was located towards the right or the left side.
    Will be used in the function auto_side.

    Calculates, in the 4 first bends, which bend was the highest in amplitude, and on which side this bout was.


    :param bout_num: Bout number
    :param df_bout: DataFrame object with parameters of each bout
    :param df_frame: DataFrame object with parameters of each frame

    :return: Value of the highest bend whithin the 4 first of a bout.
    """
    bout_start = df_bout.BoutStartVideo[bout_num]
    bout_end = df_bout.BoutEndVideo[bout_num]
    bend_indices = find_indices(df_frame.Bend_Amplitude[bout_start:bout_end], lambda e: math.isnan(e) is False)
    max_bend = np.nanmax(df_frame.Bend_Amplitude[[x + bout_start for x in bend_indices[0:4]]])
    min_bend = np.nanmin(df_frame.Bend_Amplitude[[x + bout_start for x in bend_indices[0:4]]])
    print('bout', bout_num)
    if abs(max_bend) > abs(min_bend):
        decisive_bend = max_bend
    else:
        decisive_bend = min_bend
    print('decisive bend:', decisive_bend)
    return decisive_bend


def auto_side(bout_num, df_bout, df_frame, thresh):
    """
    Calculates automatically the side towards which a bout was directed, using decisive_bend.

    Decides a bout was located on the right side if decisive bend (eg highest amp bout in the 4 first)
    is higher than threshold in degrees (arbitrary value set by me.)

    Decides a bout was located on the left side if decisive bend (eg highest amp bout in the 4 first)
    is lower than threshold in degrees (arbitrary value set by me.)

    NB: negative value of tail angle corresponds to the tail being on the right side of the fish,
    and reciprocally for positive value and left side.

     :param bout_num: Bout number
    :param df_bout: DataFrame object with parameters of each bout
    :param df_frame: DataFrame object with parameters of each frame
    :return: side of the bout
    """
    if decisive_bend(bout_num, df_bout, df_frame) > thresh:
        output = 'L'
    elif decisive_bend(bout_num, df_bout, df_frame) < -thresh:
        output = 'R'
    else:
        output = 'F'
        print('bout', bout_num)
        print('Warning: was categorized as turns, but max amplitude of bout is below', thresh)
        print('Bout was removed from TURN and tagged as FORWARD')
    return output


def replace_category(bout, df_bouts, df_frame, thresh):
    """
    This function reads, for a specified bout, the category it was given by the bout clustering (a int)
    and replaces it with a user readable format: F (forward), L (left turn), R (right turn), O (others).

    It is based on how the clustering works at the time where this function is written, and could require some
    rewriting if the clustering doesn't give categories with the same number.

    Cat 0 corresponds to turns: in order to determine if it was left or right turn, the function calls function
    auto_side.
    Cat 1 corresponds to forward.
    Cat 2 corresponds to others.
    If a bout was not classified, the resulting category is NaN.

    :return the str category of the given bout.
    """
    cat = df_bouts.classification[bout]
    if cat == 1:
        str_cat = 'F'
    elif cat == 0:
        str_cat = auto_side(bout, df_bouts, df_frame, thresh)
    elif cat == 2:
        str_cat = 'O'
    else:
        str_cat = 'Exc'
    return str_cat


def replace_category2(bout, df_bouts, df_frame, thresh, struggle_lim):
    """
    This function reads, for a specified bout, the category it was given by the bout clustering (a int)
    and replaces it with a user readable format: F (forward), L (left turn), R (right turn), O (others).

    It is based on how the clustering works at the time where this function is written, and could require some
    rewriting if the clustering doesn't give categories with the same number.

    This one works with a clustering based on 2 clusters. You end up with

    Cat 0 corresponds to forward
    Cat 1 corresponds to turns: in order to determine if it was left or right turn, the function calls function
    auto_side.
    Cat 2 corresponds to all the bouts that were flagged, therefore not put in cluster.
    If a bout was not classified, the resulting category is NaN.

    :return the str category of the given bout.
    """
    cat = df_bouts.classification[bout]
    if cat == 0:
        if abs(df_bouts.Max_Bend_Amp[bout]) > struggle_lim:
            str_cat = 'S'
            print('Bout was categorized as forward, but max amplitude was higher than:', struggle_lim)
            print('Excluded from F and tagged as STRUGGLE')
        else:
            str_cat = 'F'
    elif cat == 1:
        if abs(df_bouts.Max_Bend_Amp[bout]) > struggle_lim:
            str_cat = 'S'
            print('Bout was categorized as turn, but max amplitude was higher than:', struggle_lim)
            print('Excluded from TURNS and tagged as STRUGGLE')
        else:
            str_cat = auto_side(bout, df_bouts, df_frame, thresh)
            if str_cat == 0:
                print('bout', bout)
                print('Warning: was categorized as turns, but max amplitude of bout is below', thresh)
                print('check angle trace.')
                str_cat = np.nan
    else:
        str_cat = 'Exc'
    return str_cat


def behavior_resampled(TA, old_fq, new_fq, nFrames, path):
    """Build a numpy array with the tail angle resampled to match the acqusition rate of the 2P microscope.
    saved in the specified path.
    This output can be loaded into suite2P for easy vizualisation.

    :param TA: tail angle array as generated by the script
    :param old_fq: type string, the time index of the behavior video in a time series format (example "0.002S")
    :param new_fq: type string, the time index of the 2P in a time series format ("0.200S")
    :param F: whatever array which length is the number of frames
    :param path: path to save this numpy array

    :return TA array downsampled
    """
    ta = pd.DataFrame(TA[:,1], index=pd.date_range(start = "00:00", periods=len(TA[:,1]), freq=old_fq))
    ta_downsample = ta.resample(new_fq).sum()
    output = np.zeros(nFrames)
    output[0:len(ta_downsample)] = ta_downsample[0]
    np.save(path+'behavior_resampled.npy', output)
    print('Saved behavior trace as numpy array to be visualized in suite2p in', path)
    return output


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
        if fq:
            time_indices = np.array(range(cls.start, cls.end))/fq
            plt.plot(time_indices, cls.ta)
            plt.plot(time_indices-(1/fq), cls.bends, 'rx', markersize=1.5)
            plt.xlabel('Time [s]')
        else:
            plt.plot(df_frames.cls.ta)
            plt.plot(df_frames.cls.bends, 'rx', markersize=1.5)
            plt.xlabel('Frame')
        plt.ylim(-50, 50)
        plt.title('Tail angle over time for bout ' + str(cls.num))
        plt.ylabel('Tail angle [°]')

    def view_bout_video(cls, video_path):
        cap = cv2.VideoCapture(video_path + cls.num + '.avi')
        if cap.isOpened() is False:
            print("Error opening video stream or file")

        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                cv2.imshow('Frame', frame)
                # define how long to wait before frames to be displayed
                # press q if you want to quit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()
    # TODO: think about other functions to add to this class.


def create_bout_objects(df_bouts, df_frames, output_path):
    """Creates a pandas Series with Bout object for each bout of the df_bouts.
    Saves it in pickle format in the desired direction."""
    bouts = pd.Series([Bout(df_bouts, df_frames, i) for i in df_bouts.index])
    bouts.to_pickle(output_path + 'bouts')
    return bouts

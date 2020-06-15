import os, datetime
import pandas as pd


class Exp:
    """Class for the experiment of one fish"""

    def __init__(self, path, fishlabel, csv_path):
        # date of experiment
        try:
            self.date = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        except FileNotFoundError:
            self.date = fishlabel.split('_')[0]

        print('date fo experiment:', self.date)

        # get parameters of experiment, either in the csv summary file or by user
        csv_file = pd.read_csv(csv_path)
        fishinfo = csv_file[csv_file['Fishlabel'] == fishlabel]

        # number of planes taken for this fish
        nPlanes = 0
        try:
            for i in os.listdir(path):
                if os.path.isdir(os.path.join(path, i)):
                    nPlanes += 1
        except FileNotFoundError:
            nPlanes = int(fishinfo['nPlanes'])
        print('Nplanes is ', nPlanes)
        a = str(input('Ok ? press anything for no'))
        if a:
            nPlanes = str(input('enter NPlanes:'))
        self.nPlanes = nPlanes

        # frame rate of behavior acquisition
        fps_beh = fishinfo['FrameRateCamera']
        print('fps behavior is ', int(fps_beh))
        a = str(input('Ok ? press anything for no'))
        if a:
            fps_beh = str(input('enter camera speed for behavior:'))
        self.fps_beh = float(fps_beh)

        # frame rate of 2P acquisition
        fps_2p = fishinfo['FrameRate2P']
        print('fps 2p is ', float(fps_2p))
        a = str(input('Ok ? press anything for no'))
        if a:
            fps_2p = str(input('enter camera speed for 2p:'))
        self.fps_2p = float(fps_2p)
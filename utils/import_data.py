import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(1, '../utils/')
import pickle


class ReAnalyze(Exception):
    pass


def load_experiment(path, fishlabel):
    """Loads the experiment object corresponding the the Exp class. This object contains all the info about the
    experiment performed on the fish adressed by fishlabel."""
    with open(path + 'exps/' + fishlabel + '_exp', 'rb') as handle:
        experiment = pickle.load(handle)
    return experiment


def load_analysis_log(path, fishlabel, trial):
    try:
        with open(path + 'logs/' + fishlabel + '_' + trial + '_analysis_log', 'rb') as handle:
            analysis_log = pickle.load(handle)
    except FileNotFoundError:
        print('Did not find analysis log file in ', path + 'logs/' + fishlabel + '_' + trial + '_analysis_log')
        analysis_log = np.nan
    return analysis_log


def load_trials_correspondence(output_path, fishlabel):
    trials_correspondence = pd.read_pickle(output_path + 'dataset/' + fishlabel + '/resume_fish_' + fishlabel)
    print(trials_correspondence)
    return trials_correspondence


def load_behavior_dataframe(fishlabel, trial, path):
    frame_dataset = pd.read_pickle(
        path + 'dataset/' + fishlabel + '/' + trial + '/analyzed_frame_dataset')
    bout_dataset = pd.read_pickle(
        path + 'dataset/' + fishlabel + '/' + trial + '/analyzed_bout_dataset')
    return frame_dataset, bout_dataset


def load_behavior_dataframe_plane(fishlabel, depth, path):

    df_frame_plane = pd.read_pickle(path + 'dataset/' + fishlabel + '/df_frame_plane_' + depth)
    df_bout_plane = pd.read_pickle(path + 'dataset/' + fishlabel + '/df_bout_plane_' + depth)

    return df_frame_plane, df_bout_plane
    

def load_suite2p_outputs(fishlabel, trial, input_path):
    """Load every output that the suite2p gives you
    Arguments given are fishlabel, real_trial_num and folder_path.
    If folder_path is not given, automatically check for the data path in the summary csv file.
    You can change the path to the summary csv file here in the function.
    If folder_path is give,;
    Returns F, Fneu, spks, stat, ops, iscell"""
    if not os.path.exists(input_path):
        raise FileNotFoundError('Path to your folder is not valid.')
    try:
            F = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/F.npy', allow_pickle=True)
            Fneu = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/Fneu.npy', allow_pickle=True)
            spks = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/spks.npy', allow_pickle=True)
            stat = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/stat.npy', allow_pickle=True)
            ops = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/ops.npy', allow_pickle=True)
            ops = ops.item()
            iscell = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/iscell.npy', allow_pickle=True)
    except FileNotFoundError:
        F = np.load(input_path + fishlabel + '/' + trial + '/F.npy', allow_pickle=True)
        Fneu = np.load(input_path + fishlabel + '/' + trial + '/Fneu.npy', allow_pickle=True)
        spks = np.load(input_path + fishlabel + '/' + trial + '/spks.npy', allow_pickle=True)
        stat = np.load(input_path + fishlabel + '/' + trial + '/stat.npy', allow_pickle=True)
        ops = np.load(input_path + fishlabel + '/' + trial + '/ops.npy', allow_pickle=True)
        ops = ops.item()
        iscell = np.load(input_path + fishlabel + '/' + trial + '/iscell.npy', allow_pickle=True)
    return F, Fneu, spks, stat, ops, iscell


def load_output_struct(fishlabel, trial, path, suite2p_path, reanalyze=False):
    try:
        with open(path + fishlabel + '/' + trial + '/struct', 'rb') as s:
            output_struct = pickle.load(s)
        if reanalyze: raise ReAnalyze
    except (FileNotFoundError, ReAnalyze):
        if reanalyze:
            print('Existing output struct, but you chose to create a new one.')
        else:
            print('No output struct was found in the specified path.')
        print('Creating new output struct')
        output_struct = {'F': [], 'Fneu': [], 'spks': [], 'stat': [], 'ops': [], 'iscell': [],
                         'F_corrected': [], 'DFF': [], 'DFF_filtered': [], 'cells_index': [], 'noise': [],
                         'noise_filtered': [],
                         'signal_noise': [], 'threshold': [], 'motion_artifact': [], 'time_indices': [], 'TA_all': [],
                         'df_bouts': [], 'nROI': [], 'nFrames': []}
        F, Fneu, spks, stat, ops, iscell = load_suite2p_outputs(fishlabel, trial, suite2p_path)
        output_struct['F'] = F
        output_struct['Fneu'] = Fneu
        output_struct['spks'] = spks
        output_struct['stat'] = stat
        output_struct['ops'] = ops
        output_struct['iscell'] = iscell
        # define number of frames for calcium imaging recording
        nROI, nFrames = F.shape
        output_struct['nROI'] = nROI
        output_struct['nFrames'] = nFrames
    return output_struct


def load_config_file(fishlabel, index, path):
    try:
        with open(path + fishlabel + '/config_file', 'rb') as f:
            config_file = pickle.load(f)
    except FileNotFoundError:
        print('Missing analysis')
        quit()

    nTrials, frame_rate_2p, fq = len(config_file['Trial_num']), config_file['Frame_rate_2p'][index], \
                                 config_file['Fq_camera'][index]
    return config_file, nTrials, frame_rate_2p


def load_bout_object(path, fishlabel, trial):
    with open(path + 'dataset/' + fishlabel + '/' + trial + '/bouts', 'rb') as handle:
        bouts = pickle.load(handle)
    return bouts
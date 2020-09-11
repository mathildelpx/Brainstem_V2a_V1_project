import pandas as pd
import numpy as np
from utils.combine_trials import combineTrials, createTailAngleArrayPlane
from utils.import_data import *
from tools.list_tools import *
from utils.functions_behavior import *


fishlabel = '200813_F1'
summary_file_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/summaryDataFinal.csv'
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'
# As we use the code used for experiments of multiple wells,
# we have to call the first well (only one in the experiments
# for which this script was written.
numWell = 0

try:
    summary_file = pd.read_csv(summary_file_path,
                               header=0)
except FileNotFoundError:
    print('The path to the summary file is not valid.')
    quit()

summary_fish = summary_file[summary_file['Fishlabel'] == fishlabel]

# get frame rate of the 2P microscope from the summary file
experiment = load_experiment(output_path, fishlabel)
experiment = experiment[0]

planes = get_unique_elements(summary_fish['Depth'])

pd.Series(list(planes)).apply(combineTrials,
                              args=(summary_fish, fishlabel, output_path))
pd.Series(list(planes)).apply(createTailAngleArrayPlane, args=(summary_fish, fishlabel, output_path))

for depth in planes:
    # Create a list of objects from class Bout and save it
    with open(output_path + 'dataset/' + fishlabel + '/df_frame_plane_' + depth, 'rb') as f:
        df_frame = pickle.load(f)
    with open(output_path + 'dataset/' + fishlabel + '/df_bout_plane_' + depth, 'rb') as f:
        df_bouts = pickle.load(f)
    create_bout_objects(df_bouts, df_frame, output_path + 'dataset/' + fishlabel + '/' + depth + '/')

# create_bout_objects(df_bouts_plane, df_frame_plane, output_path + 'dataset/' + fishlabel + '/' + depth + '/')
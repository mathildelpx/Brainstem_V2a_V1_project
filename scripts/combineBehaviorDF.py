import pandas as pd
import numpy as np
import time
import logging
from utils.combine_trials import combineTrials, createTailAngleArrayPlane
from utils.import_data import *
from tools.list_tools import *
from utils.functions_behavior import *


fishlabel = '200813_F1'
summary_file_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/summaryDataFinal.csv'
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'

if __name__ == '__main__':
    try:
        os.mkdir('../logs/'+fishlabel)
    except FileExistsError:
        pass

    logging.basicConfig(filename='../logs/'+fishlabel+'/'+os.path.basename(__file__)+'.log',
                        level=logging.DEBUG)
    logging.info('Analysing fish' + fishlabel)
    logging.info('Summary file used: ' + summary_file_path)
    logging.info('output path used: ' + output_path)
    logging.info('Time at which this log started: ' + time.strftime())


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
                              args=(summary_fish, fishlabel, output_path, experiment))

if __name__ == '__main__':
    logging.info('Combined the trials for all planes '+planes)

pd.Series(list(planes)).apply(createTailAngleArrayPlane, args=(summary_fish, fishlabel, output_path))

if __name__ == '__main__':
    logging.info('Created tail angle array for all planes '+planes)

for depth in planes:
    # Create a list of objects from class Bout and save it
    with open(output_path + 'dataset/' + fishlabel + '/df_frame_plane_' + depth, 'rb') as f:
        df_frame = pickle.load(f)
    with open(output_path + 'dataset/' + fishlabel + '/df_bout_plane_' + depth, 'rb') as f:
        df_bouts = pickle.load(f)
    create_bout_objects(df_bouts, df_frame, output_path + 'dataset/' + fishlabel + '/' + depth + '/')

    if __name__ == '__main__':
        logging.info('Created bouts objects for depth ' + depth)

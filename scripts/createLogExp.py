import os, datetime
import pandas as pd
import pickle
from utils.class_lib import Exp

fishlabel = '200930_F1'

data_path = '/network/lustre/iss01/wyart/rawdata/2pehaviour/' + fishlabel + '/2P/'
csv_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/summaryDataFinal.csv'
save_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/exps/'

experiment = pd.Series(Exp(data_path, fishlabel, csv_path))

try:
    os.mkdir(save_path)
except FileExistsError:
    pass

experiment.to_pickle(save_path + fishlabel + '_exp')

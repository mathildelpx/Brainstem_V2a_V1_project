import os, datetime
import pandas as pd
import pickle
from utils.class_lib import Exp


fishlabel = '191022_F1'

data_path = '/network/lustre/iss01/wyart/rawdata/2pehaviour/' + fishlabel + '/Calcium_imaging/'
csv_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/summaryData_MartinMathilde.csv'
save_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/exps/'

experiment = pd.Series(Exp(data_path, fishlabel, csv_path))

try:
    os.mkdir(save_path)
except FileExistsError:
    pass

experiment.to_pickle(save_path + fishlabel + '_exp')

import json
import pyabf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import functions_ZZ_extraction as fct

fishlabel = '200903_F1'
ZZ_output_path = '/home/mathilde.lapoix/anaconda3/lib/python3.7/site-packages/zebrazoom/ZZoutput/'
output_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/ML_pipeline_output/'
depth = '215um'
trials = [0, 1, 3]
colors = ['orange', 'magenta', 'green', 'grey']
labels = ['stim (from R to L)', 'stim (from L to R)', 'stim (from B to F)', 'no stim']

# set up env
try:
    os.mkdir(output_path+'fig/'+fishlabel)
    os.mkdir(output_path+'np_array/'+fishlabel)
    os.mkdir(output_path + 'fig/' + fishlabel + '/' + depth)
    os.mkdir(output_path + 'np_array/' + fishlabel + '/' + depth)
except FileExistsError:
    try:
        os.mkdir(output_path+'fig/'+fishlabel+'/'+depth)
    except FileExistsError:
        pass
    try:
        os.mkdir(output_path+'np_array/'+fishlabel+'/'+depth)
    except FileExistsError:
        pass


# import abf ref file
abf = pyabf.ABF("/network/lustre/iss01/wyart/rawdata/2pehaviour/200813_F1/ePhys/200813_F1__0004.abf")

# access OMR stim
abf.setSweep(sweepNumber=0, channel=1)

# the abf file only shows you the data for one run, so you need to repeat the stim trace for the x number of runs
x_OMR = np.arange(0,2*(abf.sweepX[-1]+0.0005), 0.0005)
y_OMR = np.tile(abf.sweepY,2)


for trial in trials:
    filename = fishlabel + '_' + depth + '-' + str(trial)
    txt_file = ZZ_output_path + filename + '/results_' + filename + '.txt'
    with open(txt_file) as f:
        struct = json.load(f)
        substruct = struct["wellPoissMouv"][0][0]

    NBout = len(substruct)
    End_Index = substruct[NBout - 1]["BoutEnd"]
    index_frame = pd.Series(range(End_Index + 1))
    TA = pd.Series(index_frame).apply(fct.tail_angle, args=(substruct, NBout))

    # x_TA in seconds
    x_TA = np.arange(0,len(TA))/300

    plt.figure(trial)
    plt.plot(x_OMR, y_OMR, color=colors[trial], label=labels[trial])
    plt.plot(x_TA, TA, color='dodgerblue', label='TA')
    plt.ylim((-110, 110))
    plt.xlabel(abf.sweepLabelX)
    plt.ylabel(abf.sweepLabelY + ' / Tail angle (°)')
    plt.legend()
    plt.savefig(output_path+'fig/'+fishlabel+'/'+depth+'/OMRvsTA_trial'+str(trial)+'.png')

    # create resampled ta_array
    TA_array = np.array(TA)
    ta = pd.DataFrame(TA_array, index=pd.date_range(start="00:00:00", periods=len(TA), freq="0.003333S"))
    ta_resampled = ta.resample("0.207S").sum()
    nFrames = int(input('Number of 2P frames for trial ' + str(trial)))
    if len(ta_resampled != nFrames):
        to_fill = int(nFrames - len(ta_resampled))
    ta_resampled = ta_resampled.append(pd.DataFrame(np.repeat(0, to_fill)), ignore_index=True)
    if trial == 0:
        dict_ta = {}
    dict_ta[str(trial)] = ta_resampled

overall_ta = dict_ta['0'].append(dict_ta['1'], ignore_index=True)
# overall_ta = overall_ta.append(dict_ta['2'], ignore_index=True)
# if len(trials) > 3:
#     overall_ta = overall_ta.append(dict_ta['3'], ignore_index=True)

plt.close()
plt.figure()
plt.plot(overall_ta)
np.save(output_path+'np_array/'+fishlabel+'/'+depth+'/overall_ta.npy', np.array(overall_ta))

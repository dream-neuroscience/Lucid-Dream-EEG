# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:36:51 2020

@author: caghangir
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import extremeEEGSignalAnalyzer as chetto_EEG
import pickle
chetto_EEG = chetto_EEG.extremeEEGSignalAnalyzer()
%matplotlib qt
os.chdir('C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Codes/Pickle Files')
#%%
folder_path = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/FILD_Raw_data/Lucidity Found'
files = list()
for file in os.listdir(folder_path):
    if file.endswith(".edf"):
        files.append(file)
            
#%%===== Definitions =====
allFs = list()
dataChannels = list()
dataSamplingRates = list()
lengthSeconds = list()
count = len(files)
dataInfos = list()
tempList = list()
dataSets = list()
#===== Definitions =====
#%%======= Read EDF File =============
for i in range(count):
    data = mne.io.read_raw_edf(folder_path + '/' + files[i])
    dataSets.append(data)
    dataInfo = data.info
    dataInfos.append(dataInfo)
    dataChannels.append(dataInfo['ch_names'])
    dataSamplingRates.append(int(dataInfo['sfreq']))
    lengthSeconds.append(len(list(data[0])[0].flatten()) / dataInfo['sfreq'])
    
    allFs.append(dataInfo['sfreq'])
    # allDataChannels.append(dataInfo['ch_names'])
#%%======= Decode EDF Data =============
T = 30 #secs
len_epoch   = int(allFs[0] * T)
n_channels = len(dataChannels[0])

chetto_EEG = chetto_EEG.extremeEEGSignalAnalyzer()

raw_data_0 = dataSets[0].get_data()
raw_data_0 = chetto_EEG.butter_bandpass_filter(data=raw_data_0, lowcut=0.1, highcut=30, fs=dataSamplingRates[0])
raw_data_0 = raw_data_0[:, 0:raw_data_0.shape[1] - raw_data_0.shape[1] % len_epoch] #cut the tail

#====== Reshape ======
# raw_data_0_epoched = np.reshape(raw_data_0, (n_channels, int(raw_data_0.shape[1] / len_epoch), len_epoch), order='F')
raw_data_0_epoched = np.reshape(raw_data_0, (n_channels, len_epoch, int(raw_data_0.shape[1] / len_epoch)), order='F')
raw_data_0_epoched = np.transpose(raw_data_0_epoched, (0,2,1))
#====== Reshape ======

#==== Lucidity Markers ====
raw_data_0_epoched_EOGs = raw_data_0_epoched[2:4,:,:]

x1 = raw_data_0_epoched_EOGs[:,567,:]

plt.plot(x1[0]+400, color='blue')
plt.plot(x1[1], color='red')
#==== Lucidity Markers ====
#%% ========= Decode by Seconds ================
T = 30 #secs
len_epoch   = int(allFs[0] * T)

# chetto_EEG = chetto_EEG.extremeEEGSignalAnalyzer()

raw_data_0 = dataSets[0].get_data()
raw_data_0 = chetto_EEG.butter_bandpass_filter(data=raw_data_0, lowcut=0.1, highcut=30, fs=dataSamplingRates[0])
raw_data_0 = raw_data_0[:, 0:raw_data_0.shape[1] - raw_data_0.shape[1] % len_epoch] #cut the tail

#====== Seconds =========
raw_data_0_EOGs = raw_data_0[0:2,:]
seconds = np.array([13429.5, 13432]) * allFs[0]
seconds = seconds.astype(int)
ld_marks_0 = raw_data_0_EOGs[:,seconds[0]:seconds[1]]
#====== Seconds =========

#==== Plot =====
time_interval = len(ld_marks_0[0]) / allFs[0]
time_interval = np.linspace(0, time_interval, num=len(ld_marks_0[0]))  # seconds

plt.figure()
plt.plot(time_interval, ld_marks_0[0] + (max(np.abs(ld_marks_0[0])) - min(np.abs(ld_marks_0[1]))), color='blue', linewidth=2)
plt.plot(time_interval, ld_marks_0[1], color='red', linewidth=2)
plt.title('LRLR Marker', size=30)
plt.legend(['EOG-0', 'EOG-1'], prop={'size': 20})
plt.xlabel('Time [Seconds]', size=25)
plt.ylabel('Amplitude [uV]', size=25)

#=== Maximize ====
figure = plt.gcf()  # get current figure
figure.set_size_inches(32, 18)
#=== Maximize ====
plt.show()
#==== Plot =====

#%%=========== Inputs ==================
folder_path_o_kniebeugen = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/2_Christian_Tausch/Final EDF Files'
folder_path_FILD = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/FILD_Raw_data/Lucidity Found'
folder_path_Jarrot = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/LD_EEG_Jarrot/EDF Files'
folder_path_Sergio = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/Sergio'

#================================
LRLR_timeIntervals_o_kniebeugen = list() #input as seconds
LRLR_timeIntervals_o_kniebeugen.append(np.array([[17023.5, 17026.5],[17036.5, 17039],[17071,17073.5],[17082,17084.5]])) #203
LRLR_timeIntervals_o_kniebeugen.append(np.array([[22969,22971],[22990.5,22992],[23057,23060.5],[23067.5,23070],[23076.5,23079]])) #301
LRLR_timeIntervals_o_kniebeugen.append(np.array([[23113.5,23115],[23117,23118.5],[23125.5,23130],[23303.5,23307],[28753,28754.5],
                                                 [28764.5,28767],[28770,28774],[28812,28815],[28821,28822.7],[28911.5,28915.5]])) #302
LRLR_timeIntervals_o_kniebeugen.append(np.array([[16174.5,16187],[16221.5,16230],[16238.5,16239.5],[16268,16269],[22485,22490],[22512.5,22521.5],\
                                                 [22556.5,22558],[22562,22563.5],[22567,22571],[22612,22613.5],[22629,22631],[22639,22641],[22671,22673],\
                                                 [22696,22700]])) #303
LRLR_timeIntervals_o_kniebeugen.append(np.array([[28404.5,28405.5],[28408,28410.5],[28440,28441.5],[28445.5,28449.5],[31929,31930.5],[31937,31938.5],\
                                                 [31967,31968.5],[31972.5,31974],[31978.5,31983],[32034.5,32037.5],[32050,32052]])) #304
LRLR_timeIntervals_o_kniebeugen.append(np.array([[30396.5,30398.5],[30402,30403.5],[30429,30430.5],[30916.5,30921.5],[30925.5,30927]])) #406
LRLR_timeIntervals_o_kniebeugen.append(np.array([[28197.5,28199],[28212.5,28214.5],[28219.5,28222],[28246.5,28250.5],[28258,28264.5],\
                                                [28289.5,28293],[28299.5,28304]])) #501
LRLR_timeIntervals_o_kniebeugen.append(np.array([[17956,17957.5],[17980,17981],[17987,17988],[18000,18001],[18007.5,18008],[18016,18018.5],\
                                                 [24696.5,24698],[24710,24711],[24717,24718.5],[24743,24744],[24752.5,24753.5],\
                                                 [24769.5,24772],[24786.5,24788],[24817,24820],[24837.5,24839.5]])) #603
#================================
LRLR_timeIntervals_FILD = list()
LRLR_timeIntervals_FILD.append(np.array([[40397,40400],[40510,40513]])) #10
LRLR_timeIntervals_FILD.append(np.array([[39264,39267]])) #14
LRLR_timeIntervals_FILD.append(np.array([[38384,38386]])) #17
LRLR_timeIntervals_FILD.append(np.array([[41658,41660]])) #19
LRLR_timeIntervals_FILD.append(np.array([[37599,37602],[37661,37664],[37699,37703],[37763,37768],[37829,37833],[37876,37878],[42969,42972]])) #20
LRLR_timeIntervals_FILD.append(np.array([[36151.5,36156],[36215,36220]])) #21
LRLR_timeIntervals_FILD.append(np.array([[35918,35921],[35969.5,35974.5]])) #4
LRLR_timeIntervals_FILD.append(np.array([[39793,39796]])) #59
LRLR_timeIntervals_FILD.append(np.array([[34041.5,34044],[34051,34053],[34160,34164]])) #8

#================================
LRLR_timeIntervals_Jarrot = list()
LRLR_timeIntervals_Jarrot.append(np.array([[13429.5,13432],[19386,19392.5]]))
LRLR_timeIntervals_Jarrot.append(np.array([[27305,27309.5]]))
LRLR_timeIntervals_Jarrot.append(np.array([[33248,33252.5]]))
LRLR_timeIntervals_Jarrot.append(np.array([[13463.5,13467],[13564,13569],[18302,18309],[18668,18672],[24590,24594]]))
LRLR_timeIntervals_Jarrot.append(np.array([[15444,15450],[15488,15494],[26570,26575],[27020.5,27029.5],[27113,27117],\
                                           [32505,32514],[33185,33191]]))
LRLR_timeIntervals_Jarrot.append(np.array([[19490.5,19499]]))
LRLR_timeIntervals_Jarrot.append(np.array([[21120.5,21125.5],[36024.5,36028],[38747,38752],[40072.5,40078]]))

#================================
LRLR_timeIntervals_Sergio = list()
LRLR_timeIntervals_Sergio.append(np.array([[11016.5,11024.5],[11031.5,11040.5],[11059,11069]]))
#=========== Inputs ==================

#%% ======= LRLR Marker Cutting ===========
# chetto_EEG = chetto_EEG.extremeEEGSignalAnalyzer()
low_cut = 0.1 #Hz
high_cut = 49 #Hz
eog_indexes_okniebeugen = np.array([2,3])
eog_indexes_FILD = np.array([0,1])
eog_indexes_Jarrot = np.array([[0,1],[0,1],[0,1],[0,1],[10,11],[0,1],[0,1]])
eog_indexes_Sergio = np.array([24,25])
saving_directory_okniebeugen = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Plots/LRLR Markers/O_Kniebeugen'
saving_directory_FILD = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Plots/LRLR Markers/FILD'
saving_directory_Jarrot = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Plots/LRLR Markers/Jarrot'
saving_directory_Sergio = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Plots/LRLR Markers/Sergio'

o_kniebeugen_datachunk = chetto_EEG.LRLRMarker_cut_data_by_time_interval(folder_path=folder_path_o_kniebeugen, LRLR_timeIntervals=LRLR_timeIntervals_o_kniebeugen,\
                                            saving_directory=None, explanation='O_kniebeugen', \
                                            eog_indexes=eog_indexes_okniebeugen, low_cut=low_cut, high_cut=high_cut)
fild_datachunk = chetto_EEG.LRLRMarker_cut_data_by_time_interval(folder_path=folder_path_FILD, LRLR_timeIntervals=LRLR_timeIntervals_FILD,\
                                            saving_directory=None, explanation='FILD', \
                                            eog_indexes=eog_indexes_FILD, low_cut=low_cut, high_cut=high_cut)
jarrot_datachunk = chetto_EEG.LRLRMarker_cut_data_by_time_interval(folder_path=folder_path_Jarrot, LRLR_timeIntervals=LRLR_timeIntervals_Jarrot,\
                                            saving_directory=None, explanation='Jarrot', \
                                            eog_indexes=eog_indexes_Jarrot, low_cut=low_cut, high_cut=high_cut)
sergio_datachunk = chetto_EEG.LRLRMarker_cut_data_by_time_interval(folder_path=folder_path_Sergio, LRLR_timeIntervals=LRLR_timeIntervals_Sergio,\
                                            saving_directory=None, explanation='Sergio', \
                                            eog_indexes=eog_indexes_Sergio, low_cut=low_cut, high_cut=high_cut)
#%% ========== Lucidity Period Taking ===========
folder_path_o_kniebeugen = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/2_Christian_Tausch/Final EDF Files'
folder_path_FILD = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/FILD_Raw_data/Lucidity Found'
folder_path_Jarrot = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/LD_EEG_Jarrot/EDF Files'
folder_path_Sergio = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/Sergio'

#pre-lucid-post  (203,301,302,303,304_2,501,603_2)
lucidity_period_okniebeugen = np.array([[[16962, 17023.5],[17023.5, 17084.5],[17084.5, 17145]],\
                                        [[22859, 22969],[22969, 23079.5],[23079.5, 23189.5]],\
                                        [[28466, 28691],[28691, 28916],[28916, 29141]],\
                                        [[31806, 31929],[31929, 32052],[32052, 32175]],\
                                        [[22270, 22485],[22485, 22700],[22700, 22915]],\
                                        [[28091, 28197.5], [28197.5, 28304],[28304,28410.5]],\
                                        [[24603, 24696.5],[24696.5, 24839.5],[24839.5,24981]]])

#pre-lucid-post (fild_10_n1, fild_14_n2, fild_17_n2, fild_20_n1, fild_21_n2, fild_4_n1, fild_8_n2)
lucidity_period_fild = np.array([[[40140, 40396],[40397, 40587],[40587, 40620]],\
                                [[39257, 39263],[39264, 39271],[39272, 39279]],\
                                [[38313, 38383],[38383, 38463],[38463, 38543]],\
                                [[42876, 42969],[42969, 43062],[43062, 43155]],\
                                [[36033, 36151],[36151, 36344],[36344, 36537]],\
                                [[35820, 35919],[35919, 36024],[36024, 36054]],\
                                [[34020, 34041], [34041, 34168],[34170, 34200]]])

#pre-lucid-post                                   
lucidity_period_sergio = np.array([[11010,11016],[11016,11069],[11069,11126]])
#%% ============= Analysis Pipeline ==========
# file_path = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/FILD_Raw_data/Lucidity Found/FILD10_n1.edf'
# data = mne.io.read_raw_edf(file_path, preload=True)
# dataInfo = data.info
# data_channels = dataInfo['ch_names']
# ch_types = dataInfo['ch_types']

#index 2,3 : F3-A2, F4-A1
# periods = chetto_EEG.EEG_cutter_saver(file_path=file_path, channel_indexes=np.array([2,3]),\
#                                       time_interval_seconds=np.array(lucidity_period_fild[1,:,:]), low_cut=0.1, high_cut=49)       
    
periods_fild = chetto_EEG.multi_file_EEG_cutter_saver(folder_path=folder_path_FILD, channel_indexes=None,\
                             time_interval_seconds=lucidity_period_fild, low_cut=0.1, high_cut=49,\
                             selected_data=np.array([0,1,2,4,5,6,8]))
data = periods_fild['FILD10_n1']
data = [data[i][0:4,:] for i in range(3)]

#=== Overlapped Windowing / Event Data Creation =======
window_size = 100 * 4 #seconds
overlapping_size = 100 * 2  #seconds
total_data_windows = np.empty(shape=[4,0,400])
total_labels = np.empty(shape=[0])
events = np.empty(shape=[0,3])
global_index = 0
for j in range(len(data)): #number of period
    window_amount = int((len(data[j][0]) - (len(data[j][0]) % overlapping_size + window_size)) / overlapping_size + 1)
    temp_data_windows = np.zeros((4, window_amount, window_size))
    temp_labels = np.ones(window_amount) * j
    
    temp_index = 0
    temp_events = np.zeros((window_amount,3))
    for i in range(window_amount):
        temp_data_windows[:,i] = data[0][:,temp_index : temp_index + window_size]
        print(temp_index)
        temp_events[i,0], temp_events[i,2] = global_index, j
    
        temp_index += overlapping_size
        global_index += overlapping_size
               
    events = np.row_stack((events, temp_events))
    total_data_windows = np.concatenate((total_data_windows, temp_data_windows), axis=1) ##3D array concatenate axis=1
    total_labels = np.append(total_labels, temp_labels)
    
epochs_data = np.transpose(total_data_windows, (1,0,2)) #swap dimensions
#=== Overlapped Windowing / Event Data Creation =======

#======= Epoch Array ==========
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
info = mne.create_info(ch_names=['EOG-0','EOG-1','F3-A2','F4-A1'], sfreq=100, ch_types='eeg')
epochs = mne.EpochsArray(epochs_data, info=info, events=events.astype(int), event_id=event_id, baseline=(0, 0.2))
print(epochs.info)
print(epochs)
epochs.plot(scalings=50, show=True, block=False, n_epochs=10, title='Overlapping Events')
#======= Epoch Array ==========

#%% ======= MAT file reading ==============
from scipy.io import loadmat
file_path = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/2_Christian_Tausch/O_Kniebeugen/squads_9LD'
squads_9LD = loadmat(file_path)
file_path = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/2_Christian_Tausch/LD_rawdata'
LD_rawdata = loadmat(file_path)
#%% ================================= Epochs Analysis =======================================
epochs.plot(scalings=50, show=True, block=False, n_epochs=10, title='Overlapping Events', picks=['eeg','eog']) #plot epoch raw
epochs.plot_image(picks=['F3','F4']) #Epoch time-power image

#%%==== PSD of EEG, EOG Channels of all epochs=====
plt.figure()
ax = plt.axes()
epochs.plot_psd(ax=ax, dB=True, picks=['eeg','eog'], xscale='linear', estimate='power')
ax.set_title(label='PSD of Different Channels', size=25)
ax.set_xlabel(xlabel='Frequency (Hz)', size=20)
ax.set_ylabel(ylabel='µV^2/Hz (dB)', size=20)
# ax.legend(ax.lines[2::3], stages, prop={'size': 20, 'weight':3})
ax.grid(linewidth=1.2) #change grid line width
ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes

for line in plt.gca().lines: #change linewidth of axes plotted lines
    line.set_linewidth(3.)
#==== PSD of EEG, EOG Channels of all epochs=====

#%%===== PSD Among Different Periods ======
stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # random color code generation
stages = sorted(event_id.keys())

plt.figure()
ax = plt.axes()
for stage, color in zip(stages, stage_colors):
    epochs[stage].plot_psd(area_mode=None, color=color, ax=ax, fmin=0.1, fmax=49, show=False,\
                           average=True, spatial_colors=False, picks=['eeg'], db=True, estimate='power') #EOG-1, EOG-2
        
ax.set_title(label='PSD of Lucidity vs. REM vs. Awake', size=25)
ax.set_xlabel(xlabel='Frequency (Hz)', size=20)
ax.set_ylabel(ylabel='µV^2/Hz (dB)', size=20)
ax.legend(ax.lines[2::3], stages, prop={'size': 20, 'weight':3})
ax.grid(linewidth=1.2) #change grid line width
ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes

for line in plt.gca().lines: #change linewidth of axes plotted lines
    line.set_linewidth(3.)
#===== PSD Among Different Periods ======

#%%==== Multitaper Transform Between Different Periods =======
freqs = np.arange(5., 49., 0.1)
n_cycles = freqs / 2
vmin, vmax = -3., 3.  # Define our color limits.
time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)

for i in range(len(stages)):
    power = mne.time_frequency.tfr_multitaper(epochs[stages[i]], freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, \
                                              return_itc=False, picks=['F3','F4'])
    plt.figure()
    ax = plt.axes()
    power.plot([0], baseline=(0., 0.1), mode='mean', vmin=vmin, vmax=vmax, axes=ax)
    ax.set_title(label='Multitaper Spectrogram Frontal, stage :' + stages[i], size=25)
    ax.set_xlabel(xlabel='Time (s)', size=20)
    ax.set_ylabel(ylabel='Frequency (Hz)', size=20)
    ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes    
#==== Multitaper Transform Between Different Periods =======
#%% ======== Raw concatenation SSP ==========mne
from mne.io import concatenate_raws

raw_0 = chetto_EEG.read_edf_file(file_path)
raw_1 = chetto_EEG.read_edf_file(file_path_2)
raw_2 = chetto_EEG.read_edf_file(file_path_3)

raw_all = concatenate_raws([raw_0, raw_1, raw_2])
del raw_0, raw_1, raw_2

eog_projs, _ = mne.preprocessing.compute_proj_eog(raw_all, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True)
ecg_projs, _ = mne.preprocessing.compute_proj_ecg(raw_all, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True)

raw_all = chetto_EEG.sensor_location_update(raw_all, renamed_channels, channel_types, picked_channels)
raw_all.notch_filter(49, filter_length='auto', phase='zero')
raw_all.filter(l_freq=0.1, h_freq=48)

eog_projs, _ = mne.preprocessing.compute_proj_eog(raw_all, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True)
ecg_projs, _ = mne.preprocessing.compute_proj_ecg(raw_all, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True)

#======= Using projs for different raw data =======
raw_0 = chetto_EEG.read_edf_file(file_path)
raw_0 = chetto_EEG.sensor_location_update(raw_0, renamed_channels, channel_types, picked_channels)
raw_0.notch_filter(49, filter_length='auto', phase='zero')
raw_0.filter(l_freq=0.1, h_freq=48)

raw_0.info['projs'] += eog_projs + ecg_projs
raw_0.apply_proj()

#=== other one ====
raw_1 = chetto_EEG.read_edf_file(file_path_2)
raw_1 = chetto_EEG.sensor_location_update(raw_1, renamed_channels, channel_types, picked_channels)
raw_1.notch_filter(49, filter_length='auto', phase='zero')
raw_1.filter(l_freq=0.1, h_freq=48)

raw_1.info['projs'] += eog_projs + ecg_projs
raw_1.apply_proj()

#%% ======= Epoching Pipeline ========
file_path = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Datasets/FILD_Raw_data/Lucidity Found/FILD8_n2.edf'
file_path_2 = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Datasets/FILD_Raw_data/Lucidity Found/FILD10_n1.edf'
file_path_3 = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Datasets/FILD_Raw_data/Lucidity Found/FILD14_n2.edf'
renamed_channels = {'EEG F3-A2':'F3', 'EEG F4-A1':'F4', 'EOG Left':'EOG-0','EOG Right':'EOG-1', 'EMG Chin-0':'EMG',\
                    'EEG C3-A2':'C3', 'EEG C4-A1':'C4', 'EEG O1-A2':'O1', 'EEG O2-A1':'O2'}
channel_types = {'EOG-0':'eog', 'EOG-1':'eog', 'F3':'eeg', 'F4':'eeg', 'EMG':'emg', 'C3':'eeg', 'C4':'eeg',\
                 'O1':'eeg', 'O2':'eeg', 'ECG':'ecg'}    
picked_channels = ['EOG-0', 'EOG-1', 'F3', 'F4', 'C3','C4','O1','O2', 'EMG','ECG']
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
period_interval = lucidity_period_fild[6,:,:]
period_interval_2 = lucidity_period_fild[0,:,:]
period_interval_3 = lucidity_period_fild[1,:,:]

epochs_0 , events_0 = chetto_EEG.eeg_epoching_pipeline_of_given_file(file_path=file_path, renamed_channels=renamed_channels, channel_types=channel_types, \
                                            picks=['eeg','eog'], picked_channels=picked_channels, event_id=event_id, \
                                            period_interval = period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
                                            eog_projs=eog_projs, ecg_projs=ecg_projs, baseline=(0, 0.2))

epochs_1 , events_1 = chetto_EEG.eeg_epoching_pipeline_of_given_file(file_path=file_path_2, renamed_channels=renamed_channels, channel_types=channel_types, \
                                            picks=['eeg','eog'], picked_channels=picked_channels, event_id=event_id, \
                                            period_interval = period_interval_2, duration=4, overlap=2, tmin=0, tmax = None, \
                                            eog_projs=eog_projs, ecg_projs=ecg_projs, baseline=(0, 0.2))
    
# epochs_1.info['proj_id'] = 31
# epochs_1.info['proj_name'] = 'ssp'

total_epochs = chetto_EEG.epoch_concatenation(epochs_list = [epochs_0, epochs_1])
# total_epochs.apply_proj()
# chetto_EEG.PSD_of_all_stages(epochs=epochs_0, event_id=event_id)
#%% ====== PSD Comparison =====
# freqs, psd = periodogram(x=epochs_data, fs=Fs, nfft=nfft)
psd, freqs = psd_array_multitaper(x=epochs_data, sfreq=Fs, adaptive=True, normalization='full', verbose=0, fmin=0.1, fmax=48)
psd = 10 * np.log10(psd)
# plt.figure()
plt.plot(freqs, psd[0,0])
#%% ====== Epoch Concatenation FILD ======
folder_path = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Datasets/FILD_Raw_data/Lucidity Found'
renamed_channels = {'EEG F3-A2':'F3', 'EEG F4-A1':'F4', 'EOG Left':'EOG-0','EOG Right':'EOG-1', 'EMG Chin-0':'EMG',\
                    'EEG C3-A2':'C3', 'EEG C4-A1':'C4', 'EEG O1-A2':'O1', 'EEG O2-A1':'O2'}
channel_types = {'EOG-0':'eog', 'EOG-1':'eog', 'F3':'eeg', 'F4':'eeg', 'EMG':'emg', 'C3':'eeg', 'C4':'eeg',\
                 'O1':'eeg', 'O2':'eeg', 'ECG':'ecg'}    
picked_channels = ['EOG-0', 'EOG-1', 'F3', 'F4', 'C3','C4','O1','O2', 'EMG','ECG']
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
period_interval = lucidity_period_fild
fild_epochs, fild_events = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, renamed_channels=renamed_channels, \
                           channel_types=channel_types, picks=['eeg','eog'], picked_channels=picked_channels, event_id=event_id, f_min=0.1, f_max=48,
                           f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, baseline=(0, 0.2))
pickle.dump([fild_epochs, fild_events], open('fild_epochs_events','wb'))
#%% ====== Epoch Concatenation O_kniebeugen ======
folder_path_o_kniebeugen = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/2_Christian_Tausch/Final EDF Files'
renamed_channels = {'EEG F3-A2':'F3', 'EEG F4-A1':'F4', 'EOG Left':'EOG-0','EOG Right':'EOG-1', 'EMG Chin-0':'EMG',\
                    'EEG C3-A2':'C3', 'EEG C4-A1':'C4', 'EEG O1-A2':'O1', 'EEG O2-A1':'O2'}
channel_types = {'EOG-0':'eog', 'EOG-1':'eog', 'F3':'eeg', 'F4':'eeg', 'EMG':'emg', 'C3':'eeg', 'C4':'eeg',\
                 'O1':'eeg', 'O2':'eeg', 'ECG':'ecg'}    
picked_channels = ['EOG-0', 'EOG-1', 'F3', 'F4', 'C3','C4','O1','O2', 'EMG','ECG']
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
period_interval = lucidity_period_okniebeugen
o_kniebeugen_epochs, o_kniebeugen_events = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, \
renamed_channels=renamed_channels, channel_types=channel_types, picks=['eeg','eog'], picked_channels=picked_channels, \
event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
baseline=(0, 0.2)), pickle.dump([fild_epochs, fild_events], open('o_kniebeugen_epochs_events','wb'))
#%% ==== Periodograms ====
saving_directory = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Progress_Cagatay/27-06-2020/Multitaper PSD + FILD + SSP + Standardization'
periodogram_avg = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=fild_epochs, events=fild_events, fmin=0.1, fmax=48, \
n_overlap=128, nfft=1024, standardization=True, psd_type='periodogram', \
explanation='FILD, Lucid vs. REM', saving_directory=saving_directory)
    
periodogram_avg = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=fild_epochs, events=fild_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', \
explanation='FILD, Lucid vs. REM', saving_directory=saving_directory)
    
periodogram_avg = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=fild_epochs, events=fild_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='welch', \
explanation='FILD, Lucid vs. REM', saving_directory=saving_directory)
#%% ===== PSD MNE Plots ======
chetto_EEG.PSD_of_all_stages(epochs=fild_epochs, event_id=event_id, picks=['O1', 'O2'], \
                             explanation= 'FILD Dataset, Artifact Removal : SSP, Picks : O1, O2\n PSD of Lucidity vs. REM vs. Awake',\
                             saving_directory='C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Progress_Cagatay/27-06-2020')
#%% ====== PSD Lucid / REM ======
saving_directory = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Progress_Cagatay/27-06-2020/Multitaper PSD FILD + SSP + Standardization + lucid_rem_ratio'
picks=np.array([2,3,4,5,6,7])
chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=fild_epochs, events=fild_events, fmin=0.1, fmax=48, \
                                                     saving_directory=saving_directory, picks=picks, \
                                                     n_overlap=128, standardization=True, psd_type='multitaper')

#%% ============ Area 51 ========
file_path = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Datasets/FILD_Raw_data/Lucidity Found/FILD10_n1.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)
raw.notch_filter(freqs=49, filter_length='auto', phase='zero')
raw.filter(l_freq=0.1, h_freq=48)

# ============ Sensor Locations Update ==========
# === Setup layout ====
montage_list = mne.channels.get_builtin_montages() #all montages list
# layout = mne.channels.read_layout('EEG1005')
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020') #this is chosen
# === Setup layout ====

# ======== Change Channel Type & Name & Picking channels =======
raw.rename_channels(mapping={'EEG F3-A2':'F3', 'EEG F4-A1':'F4', 'EOG Left':'EOG-0','EOG Right':'EOG-1', 'EMG Chin-0':'EMG',\
                             'EEG C3-A2':'C3', 'EEG C4-A1':'C4', 'EEG O1-A2':'O1', 'EEG O2-A1':'O2'})
raw.set_channel_types(mapping={'EOG-0':'eog', 'EOG-1':'eog', 'F3':'eeg', 'F4':'eeg', 'EMG':'emg', 'C3':'eeg', 'C4':'eeg',\
                               'O1':'eeg', 'O2':'eeg', 'ECG':'ecg'})
raw.pick_channels(ch_names=['EOG-0', 'EOG-1', 'F3', 'F4', 'C3','C4','O1','O2', 'EMG','ECG'])
print(raw.info)

raw.set_montage(ten_twenty_montage) #final aim is to make it valid
# ======== Change Channel Type & Name =======

#=== Plot Sensor Location =====
#==Type 1===
layout_from_raw = mne.channels.make_eeg_layout(raw.info)
layout_from_raw.plot() # same result as: mne.channels.find_layout(raw.info, ch_type='eeg')
#==Type 1===

#==Type 2D / 3D===
fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection='3d')
raw.plot_sensors(ch_type='eeg', axes=ax2d, show_names=True)
raw.plot_sensors(ch_type='eeg', axes=ax3d, kind='3d', show_names=True)
#==Type 2D / 3D===
#=== Plot Sensor Location =====

# ============ Sensor Locations Update ==========

#=== Repair EOG/ECG artifacts with SSP (signal-space projection) =========
eog_evoked = mne.preprocessing.create_eog_epochs(raw=raw).average()
eog_evoked.apply_baseline((-0.2, 0))
eog_evoked.plot_joint()

eog_projs, _ = mne.preprocessing.compute_proj_eog(raw, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True)
ecg_projs, _ = mne.preprocessing.compute_proj_ecg(raw, n_grad=0, n_mag=0, n_eeg=1, reject=None, no_proj=True)
raw.add_proj(eog_projs)
raw.add_proj(ecg_projs)
raw.plot(scalings=50, show=True, block=False, title='EOG/ECG Projs', duration=30.0, bgcolor='white')

raw.info['projs'] += eog_projs
raw.info['projs'] += ecg_projs
raw.apply_proj()
#=== Repair EOG/ECG artifacts with SSP (signal-space projection) =========

#===== Event Creation ======
periods = lucidity_period_fild[0,:,:]
events_rem = mne.make_fixed_length_events(raw=raw, id=0, start=periods[0,0], stop=periods[0,1], duration=4, overlap=2)
events_lucid = mne.make_fixed_length_events(raw=raw, id=1, start=periods[1,0], stop=periods[1,1], duration=4, overlap=2)
events_nonrem = mne.make_fixed_length_events(raw=raw, id=2, start=periods[2,0], stop=periods[2,1], duration=4, overlap=2)
events = np.row_stack((events_rem, events_lucid, events_nonrem))
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
epochs = mne.Epochs(raw=raw, events=events, tmin=0, tmax=4 - 1. / raw.info['sfreq'], event_id=event_id, preload=True, baseline=(0, 0.2))
print(epochs.info)
#===== Event Creation ======

#== Erlacher Data ====
file_path = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Datasets/2_Christian_Tausch/Final EDF Files/O0203!_reduced.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)
raw.notch_filter(freqs=49, filter_length='auto', phase='zero')
raw.filter(l_freq=0.1, h_freq=48)


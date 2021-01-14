# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:40:39 2020

@author: caghangir
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import extremeEEGSignalAnalyzer as chetto_EEG
chetto_EEG = chetto_EEG.extremeEEGSignalAnalyzer()
import librosa
%matplotlib qt
import librosa.display

#%% ====== Read EDF =======
file_path = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/2_Christian_Tausch/Final EDF Files/O0301!_reduced.edf'
data = mne.io.read_raw_edf(file_path, preload=True)
dataInfo = data.info
data_channels = dataInfo['ch_names']
#%% ==== Plot N-channels with specific order =====
data.plot(highpass=0.1, lowpass=30, scalings=50, n_channels=6, order=np.array([2,3,0,1,4]))
#%% ======= Channel ordering data ==========
data_copy = data.copy()
channel_names = ['EOGH-0','EOGH-1','EEG_C3','EEG_C4','EMG']
data_copy = data.copy().reorder_channels(channel_names)
print(data_copy.ch_names)
#%% === Rename- channels ===
data_copy.rename_channels(mapping={'EOGH-0':'EOG-0','EOGH-1':'EOG-1'})
print(data_copy.ch_names)
#%% ==== Crop time-domain =====
cropped_data = data_copy.crop(tmin=1000, tmax=1002.5) #when you do this, data_copy will crop also itself statically
print(cropped_data)
#%% ==== Select discontinuous crops ======
cropped_data_0 = data.copy().crop(tmin=1000, tmax=1002.5)
cropped_data_1 = data.copy().crop(tmin=2000, tmax=2004.5)
cropped_data_2 = data.copy().crop(tmin=3000, tmax=3008.5)
cropped_data_0.append([cropped_data_1, cropped_data_2])

print(cropped_data_0)
print('Min times : %0.2f , Max times : %0.2f' % (cropped_data_0.times.min(), cropped_data_0.times.max()))
#%% =========== Plot PSD ==============
data_copy.plot_psd(fmax=50, fmin=0.1)
#%% ========= Band Pass Filter =========
data_bandpassFiltered = data.copy().filter(l_freq=0.1, h_freq=30)
#%% ========== EOG Artifact Detection ========
event_id = 998
eog_events = mne.preprocessing.find_eog_events(raw=data_bandpassFiltered, ch_name='EOGH-0', event_id=event_id)

picks = mne.pick_types(data_bandpassFiltered.info, meg=False, eeg=False, stim=False, eog=True, exclude='bads', include=['EOGH-0']) #you don`t need
# it just returns indexes of channel(s)

tmin, tmax = -4, 4 #seconds
epochs = mne.Epochs(data_bandpassFiltered, events=eog_events, event_id=event_id, tmin=tmin, tmax=tmax, picks=picks)
epochs_data = epochs.get_data()
print("Number of detected EOG artifacts : %d" % len(epochs_data))

#==== Plot =====
epoch_times = epochs.times
squeeze_data = np.squeeze(epochs_data).T
plt.plot(1e3 * epochs.times, np.squeeze(epochs_data).T)
plt.xlabel('Times (ms)')
plt.ylabel('EOG (µV)')
plt.show()
#==== Plot =====
#%% ======== Faking Events for Epoching ==============
T = 30 #seconds
n_epochs = int(np.floor(len(data_bandpassFiltered) / (T * dataInfo['sfreq'])))
events = np.array([np.arange(n_epochs), np.ones(n_epochs), np.ones(n_epochs)]).T.astype(int)
baseline = (None, 0.0)
tmin, tmax = -3, 3
epochs = mne.Epochs(raw=data_bandpassFiltered, events=events, tmin=tmin, tmax=tmax, baseline=baseline, picks=[2,3,0,1,4])

#==== Plot Epoch by Epoch ====
epochs.plot(block=False, scalings=50) 
#==== Plot Average =======
epochs.average()
#%% ============ Events as LRLR Marker for Epoching =========
data.rename_channels({'EOGH-0':'EOG-0','EOGH-1':'EOG-1'})
channel_names = ['EOG-0','EOG-1','EEG_C3','EEG_C4','EMG']
data.reorder_channels(channel_names)

T = 30. #seconds
event_id = {'Not Contain LRLR': 0, 'Contain LRLR': 1}
epoch_size = int(T * data.info['sfreq'])

# ======== Event Creation ==========
events =  np.zeros(shape=(8,3)).astype(int)
selected_pochs_nolrlr = np.array([567,568,582,583])
events[0,0], events[1,0], events[2,0], events[3,0] = (selected_pochs_nolrlr-1) * epoch_size 

selected_epochs_lrlr = np.array([766,767,769,770]) 
events[4,0], events[5,0], events[6,0], events[7,0] = (selected_epochs_lrlr-1) * epoch_size 
events[4,2], events[5,2], events[6,2], events[7,2] = 1,1,1,1
# ======== Event Creation ==========

# === Plot Events =====
mne.viz.plot_events(events, event_id=event_id, sfreq=data.info['sfreq']) #in order to make it sense, events must span all data
# === Plot Events =====

# ======= Epoching =======
tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included
epochs_data = mne.Epochs(raw=data, events=events, event_id=event_id, tmin=0., tmax=tmax, baseline=None)
print(epochs_data)

epochs_array = epochs_data.get_data() #numpy array of epoches
# ======= Epoching =======

# ======== Visualize PSD Among LRLR & REM =========
stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # random color code generation
stages = sorted(event_id.keys())

plt.figure()
ax = plt.axes()
for stage, color in zip(stages, stage_colors):
    epochs_data[stage].plot_psd(area_mode=None, color=color, ax=ax, fmin=0.1, fmax=30., show=False,\
                                average=True, spatial_colors=False, picks=[0,1]) #EOG-1, EOG-2
        
ax.set_title(label='PSD of Lucidity vs. REM', size=25)
ax.set_xlabel(xlabel='Frequency (Hz)', size=20)
ax.set_ylabel(ylabel='µV^2/Hz (dB)', size=20)
ax.legend(ax.lines[2::3], stages, prop={'size': 20, 'weight':3})
ax.grid(linewidth=1.2) #change grid line width
ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes

for line in plt.gca().lines: #change linewidth of axes plotted lines
    line.set_linewidth(3.)
# ======== Visualize PSD Among LRLR & REM =========
#%% ============ Plotting Sensor Locations ==========

# === Setup layout ====
montage_list = mne.channels.get_builtin_montages() #all montages list
layout = mne.channels.read_layout('EEG1005')
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020') #this is chosen
# === Setup layout ====

# ======== Change Channel Type & Name & Picking channels =======
data.rename_channels(mapping={'EEG_C3':'C3', 'EEG_C4':'C4', 'EOGH-0':'EOG-0','EOGH-1':'EOG-1'})
data.set_channel_types(mapping={'EOG-0':'eog', 'EOG-1':'eog','C3':'eeg','C4':'eeg', 'EMG':'emg'})
data.pick_channels(ch_names=['EOG-0', 'EOG-1', 'C3', 'C4', 'EMG'])
print(data.info)

data.set_montage(ten_twenty_montage) #final aim is to make it valid
# ======== Change Channel Type & Name =======

data.plot_sensors(kind='3d', ch_type='all') #3D
data.plot_sensors(ch_type='all') #3D
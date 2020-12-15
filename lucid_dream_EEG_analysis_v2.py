#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 20:27:48 2020

@author: caghangir
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import extremeEEGSignalAnalyzer as chetto_EEG
import pickle
# from mne.preprocessing import ICA
chetto_EEG = chetto_EEG.extremeEEGSignalAnalyzer()
# %matplotlib qt
os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset')

#%%=== MNE Configuration =======
try:
    mne.set_config('MNE_USE_CUDA', True)
except TypeError as err:
    print(err)
print(mne.get_config())  # same as mne.get_config(key=None)
print(mne.get_config('MNE_USE_CUDA'))
#%% ========== Lucidity Period Taking (Pre-Lucid-Post) ===========
# folder_path_o_kniebeugen = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/2_Christian_Tausch/Final EDF Files'
# folder_path_FILD = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/FILD_Raw_data/Lucidity Found'
# folder_path_Jarrot = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/LD_EEG_Jarrot/EDF Files'
# folder_path_Sergio = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/Sergio'

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

#pre-lucid-post (0008001, 000201, 000401, Luane2)                                   
lucidity_period_sergio = np.array([[[11010,11016],[11016,11069],[11069,11126]],\
                                  [[7789.2, 7802.1],[7802.1, 7815],[7815, 7827.9]],\
                                  [[27069.9, 27079.2],[27079.2, 27088.5],[27088.5, 27097.8]],\
                                  [[92.2 ,99.9],[99.9, 107.6],[107.6, 115.3]]]) 

#%% ===== Lucidity Period Taking (REM, Lucid, Wake) ============== (a bit far from each other state to make sure)
# folder_path_o_kniebeugen = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/2_Christian_Tausch/Final EDF Files'
# folder_path_FILD = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/FILD_Raw_data/Lucidity Found'
# folder_path_Jarrot = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/LD_EEG_Jarrot/EDF Files'
# folder_path_Sergio = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/Datasets/Sergio'

#rem-lucid-wake (fild_10_n1, fild_14_n2, fild_17_n2, fild_20_n1, fild_21_n2, fild_4_n1, fild_8_n2)
lucidity_period_fild = np.array([[[40020, 40210],[40397, 40587],[6126, 6316]],\
                                [[39210, 39217],[39264, 39271],[1296, 1303]],\
                                [[36750, 36830],[38383, 38463],[1800, 1880]],\
                                [[42720, 42810],[42969, 43062],[5557, 5650]],\
                                [[28680, 28873],[36151, 36344],[6059, 6252]],\
                                [[35700, 35805],[35919, 36024],[3259, 3364]],\
                                [[25920, 26040],[34041, 34168],[831, 958]]])
lengths = lucidity_period_fild[:,1,1] - lucidity_period_fild[:,1,0]

#rem-lucid-wake  (203,301,302,304_2,303,501,603_2)
lucidity_period_okniebeugen = np.array([[[16770, 16831],[17023.5, 17084.5],[51, 112]],\
                                        [[22800, 22910.5],[22969, 23079.5],[1051, 1161.5]],\
                                        [[28410, 28635],[28691, 28916],[2058, 2283]],\
                                        [[31806, 31909],[31929, 32052],[1045, 1168]],\
                                        [[22320, 22465],[22485, 22700],[618, 833]],\
                                        [[27150, 27256.5],[28197.5, 28304],[1108, 1215]],\
                                        [[23910, 24000],[24696.5, 24839.5],[909, 1052]]])
lengths = lucidity_period_okniebeugen[:,1,1] - lucidity_period_okniebeugen[:,1,0]

#rem-lucid-wake (0008001, 000401, Luane2)                                   
lucidity_period_sergio = np.array([[[3912,3965],[11016,11069],[10815,10868]],\
                                  [[24509, 24518.3],[27079.2, 27088.5],[523, 532.3]],\
                                  [[7230 ,7237],[99.9, 107.6],[12191, 12198.7]]]) 
lengths = lucidity_period_sergio[:,1,1] - lucidity_period_sergio[:,1,0]

#rem-lucid-wake (AC_P243, VR_P261_OLI_1, VR_P261_OLI_2, VR_P277)
lucidity_period_jarrod = np.array([[[19206, 19226],[19372, 19392.5],[24732, 24752]],\
                                  [[12453, 12559],[[13463.5, 13569],[18302, 18352]],[13596, 13674]],\
                                  [[35976, 36016],[33185, 33224],[5673, 5712]],\
                                  [[27513, 27801],[[21126, 21159],[40077, 40182]],[7506, 7722]]], dtype=object)
 
#rem-lucid-wake (AA00108P_1-3+, AA00109T_1-4+, AA0010AM_1-2+, AA0010AK_1-3+,
#night_2, AA0010IS_1-1+, AA0010IW_1-1+, AA0010JJ_1-1+, AA0010KD_1-1+, AA0010LX_1-1+) based on Megadata
lucidity_period_munich = np.array([[[2656, 2686],[3397, 3410],[4,17]],\
                                   [[6393, 6405],[6637, 6649],[1895, 1907]],\
                                   [[16136, 16146],[17120, 17130],[25, 35]],\
                                   [[6063, 6073],[6188, 6198],[1475, 1485]],\
                                   [[32467, 32489],[32693, 32714],[477, 498]],\
                                   [[32318, 32328],[32891, 32901],[32984, 32994]],\
                                   [[22942, 22964],[23167, 23189],[761, 782]],\
                                   [[21560, 21571],[22295,22306],[64, 75]],\
                                   [[30083,30115],[30745, 30811],[842, 908]],\
                                   [[31450, 31462],[32245, 32257],[108,120]]])
lengths = lucidity_period_munich[:,1,1] - lucidity_period_munich[:,1,0]

#fake rem-fake lucid-WAKE (AA00108P_1-2+, AA0010AW_1-2+, AA0010A9_1-2+, AA0010AK_1-2+) (subject 1,6,7,8)
lucidity_period_munic_wake = np.array([[4,17],[1895,1907],[25,35],[1475,1485]])

#%% ============== File Update ============
# file_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Final EDF Files/6_O0603!_reduced.edf'
# O603 = chetto_EEG.read_edf_file(file_path)
# renamed_channels = {'C4-A1':'EEG_C4', 'C3-A2':'EEG_C3'}
# O603.rename_channels(mapping=renamed_channels)
# O603.save('6_O0603!_reduced_updated.fif')

# main_update = [np.array(['cuk.edf','am.edf','got.edf']), [renamed_channels, renamed_channels, renamed_channels]]
# look='got.edf'
# x=np.argwhere(main_update[0] == look)[0]

#%% ====== Epoch Concatenation FILD ======
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/FILD_Raw_data/Lucidity Found'

#=== Pick Info ======
renamed_channels = {'EEG F3-A2':'F3', 'EEG F4-A1':'F4', 'EOG Left':'EOG-0','EOG Right':'EOG-1', 'EMG Chin-0':'EMG',\
                    'EEG C3-A2':'C3', 'EEG C4-A1':'C4', 'EEG O1-A2':'O1', 'EEG O2-A1':'O2'}
channel_types = {'EOG-0':'eog', 'EOG-1':'eog', 'F3':'eeg', 'F4':'eeg', 'EMG':'emg', 'C3':'eeg', 'C4':'eeg',\
                 'O1':'eeg', 'O2':'eeg', 'ECG':'ecg'}    
picked_channels = ['EOG-0', 'EOG-1', 'F3', 'F4', 'C3','C4','O1','O2', 'EMG','ECG']
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
period_interval = lucidity_period_fild
#=== Pick Info ======

ICA, SSP, CSD = False, False, False

fild_epochs, fild_events = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, renamed_channels=renamed_channels, \
                           channel_types=channel_types, picks=['eeg'], picked_channels=picked_channels, event_id=event_id, f_min=0.1, f_max=48,\
                           f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, baseline=(0, 0.2),\
                           ICA=ICA, SSP=SSP, CSD=CSD)
    
pickle.dump([fild_epochs, fild_events], open('fild_epochs_events','wb'))
#%% ====== Epoch Concatenation O_kniebeugen ======
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Final EDF Files'

#=== Pick Info ======
renamed_channels_0 = {'EEG_C3':'C3', 'EEG_C4':'C4', 'EOGH-0':'EOG-0', 'EOGH-1':'EOG-1'}
channel_types_0 = {'EOG-0':'eog', 'EOG-1':'eog', 'C3':'eeg', 'C4':'eeg', 'EMG':'emg'}
picked_channels_0 = {'EOG-0', 'EOG-1', 'C3','C4', 'EMG','ECG'}
renamed_channels = list()
channel_types = list()
picked_channels = list()
for i in range(7): renamed_channels.append(renamed_channels_0), channel_types.append(channel_types_0), \
                   picked_channels.append(picked_channels_0)
                   
renamed_channels[6] = {'C3-A2':'C3', 'C4-A1':'C4'}
renamed_channels[5] = {'EOGH1':'EOG-0', 'EOGH2':'EOG-1'}
channel_types[6] = {'C3':'eeg', 'C4':'eeg'}
picked_channels[6] = {'C3','C4', 'EMG','ECG'}

resample = [0,0,0,0,0,100,100] # 0 means no resampling
period_interval = lucidity_period_okniebeugen
#=== Pick Info ======

ICA, SSP, CSD = False, False, False

o_kniebeugen_epochs, o_kniebeugen_events = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, \
renamed_channels=renamed_channels, channel_types=channel_types, picks=['eeg'], picked_channels=picked_channels, \
event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
baseline=(0, 0.2), ICA=ICA, SSP=SSP, CSD=CSD, input_as_list=True, resample=resample)
    
pickle.dump([o_kniebeugen_epochs, o_kniebeugen_events], open('o_kniebeugen_epochs_events','wb'))
#%% ====== Epoch Concatenation Sergio ======
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Sergio/Dataset/EDF'

#=== Pick Info ======
renamed_channels = {'EEG F7':'F7', 'EEG T3':'T3', 'EEG T5':'T5', 'EEG Fp1':'Fp1', 'EEG F3':'F3','EEG C3':'C3', 'EEG P3':'P3',\
                    'EEG O1':'O1', 'EEG F8':'F8', 'EEG T4':'T4', 'EEG T6':'T6', 'EEG Fp2': 'Fp2', 'EEG F4':'F4', 'EEG C4':'C4',\
                    'EEG P4':'P4', 'EEG O2':'O2', 'EEG Fz':'Fz','EEG Cz':'Cz', 'EEG Pz':'Pz', 'EEG Oz':'Oz', 'EEG A1':'A1',\
                    'EEG A2':'A2', 'Oc1':'EOG-0', 'Oc2':'EOG-1'}
channel_types = {'EOG-0':'eog', 'EOG-1':'eog', 'F7':'eeg','T3':'eeg','T5':'eeg', 'Fp1':'eeg', 'F3':'eeg',\
                 'C3':'eeg', 'P3':'eeg', 'O1':'eeg', 'F8':'eeg', 'T4':'eeg', 'T6':'eeg', 'Fp2':'eeg', 'F4':'eeg', 'C4':'eeg',\
                 'P4':'eeg', 'O2':'eeg', 'Fz':'eeg', 'Cz':'eeg', 'Pz':'eeg', 'Oz':'eeg', 'A1':'eeg', 'A2':'eeg'}
picked_channels = list(channel_types.keys())
# picked_channels.append('EMG')
# picked_channels.append('ECG') #exclude 2 channels ('FOTO' and 'FreqC')

event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
period_interval = lucidity_period_sergio
#=== Pick Info ======

ICA, SSP, CSD = False, False, False

sergio_epochs, sergio_events = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, \
renamed_channels=renamed_channels, channel_types=channel_types, picks=['eeg'], picked_channels=picked_channels, \
event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
baseline=(0, 0.2), ICA=ICA, SSP=SSP, CSD=CSD)
    
pickle.dump([sergio_epochs, sergio_events], open('sergio_epochs_events','wb'))
#%% ====== Epoch Concatenation Jarrot ======
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/LD_EEG_Jarrot/EDF Files/Lucidity Found'

#===== Pick Info =====
renamed_channels = {'EOG1:A2':'EOG-0','EOG2:A1':'EOG-1','F3:A2':'F3','F4:A1':'F4', 'C3:A2':'C3','C4:A1':'C4','O1:A2':'O1',\
                    'O2:A1':'O2', 'ECG 2':'ECG'}
channel_types = {'EOG-0':'eog','EOG-1':'eog','F3':'eeg','F4':'eeg','C3':'eeg','C4':'eeg','O1':'eeg','O2':'eeg', 'ECG':'ecg',\
                 'EMG':'emg'}  
picked_channels = ['EOG-0', 'EOG-1', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'EMG', 'ECG', 'A1', 'A2']  
drop_channels = ['EOG1', 'EOG2','F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'EMG+', 'EMG-', 'Light', 'Battery']
eeg_reference = ['A1', 'A2']
 
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
period_interval = lucidity_period_jarrod
#===== Pick Info =====

ICA, SSP, CSD = False, False, False

jarrot_epochs, jarrot_events = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, \
renamed_channels=renamed_channels, channel_types=channel_types, picks=['eeg'], picked_channels=picked_channels, \
event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
baseline=(0, 0.2), ICA=ICA, SSP=SSP, CSD=CSD, drop_channels=drop_channels, eeg_reference=eeg_reference)
    
pickle.dump([jarrot_epochs, jarrot_events], open('jarrot_epochs_events','wb'))
#%% ====== Epoch Concatenation Lucireta ======
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Lucireta/General Data'
folder_path_wake = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Lucireta/Wake Data'

#=== Pick Info ======
renamed_channels = {'EEG F3-A2':'F3', 'EEG F4-A1':'F4', 'EOG Left':'EOG-0','EOG Right':'EOG-1', 'EMG Chin-0':'EMG',\
                    'EEG C3-A2':'C3', 'EEG C4-A1':'C4', 'EEG O1-A2':'O1', 'EEG O2-A1':'O2'}
channel_types = {'EOG-0':'eog', 'EOG-1':'eog', 'F3':'eeg', 'F4':'eeg', 'EMG':'emg', 'C3':'eeg', 'C4':'eeg',\
                 'O1':'eeg', 'O2':'eeg', 'ECG':'ecg'}    
picked_channels = ['EOG-0', 'EOG-1', 'F3', 'F4', 'C3','C4','O1','O2', 'EMG','ECG']
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
period_interval = lucidity_period_munich
#=== Pick Info ======

ICA, SSP, CSD = False, False, False

lucireta_epochs, lucireta_events = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, \
renamed_channels=renamed_channels, channel_types=channel_types, picks=['eeg'], picked_channels=picked_channels, \
event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
baseline=(0, 0.2), ICA=ICA, SSP=SSP, CSD=CSD, Lucireta=True)

pickle.dump([lucireta_epochs, lucireta_events], open('lucireta_epochs_events','wb'))
    
#Just awakes
event_id = {'Wake': 2}
lucireta_epochs_wake, lucireta_events_wake = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path_wake, \
renamed_channels=renamed_channels, channel_types=channel_types, picks=['eeg'], picked_channels=picked_channels, \
event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=lucidity_period_munic_wake, duration=4, overlap=2, tmin=0, tmax = None, \
baseline=(0, 0.2), ICA=ICA, SSP=SSP, CSD=CSD, Lucireta=True)
    
pickle.dump([lucireta_epochs, lucireta_events], open('lucireta_epochs_events','wb'))
#%% ==== Periodograms ====
saving_directory = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Progress_Cagatay/27-06-2020/Multitaper PSD + FILD + SSP + Standardization'

periodogram_avg = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=fild_epochs, events=fild_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', \
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
#%% ============ Area 51 (Main update trial) =============
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Final EDF Files/6_O0603!_reduced.edf'
raw = chetto_EEG.read_edf_file(folder_path)

#=== Main Update ==== 
ren_channel_5_O0501_reduced = {'EOGH1':'EOGH-0', 'EOGH2':'EOGH-1'}
ren_channel_6_O0603_reduced = {'C4-A1':'EEG_C4', 'C3-A2':'EEG_C3'}
main_update = [np.array(['5_O0501!_reduced.edf','6_O0603!_reduced.edf']), [ren_channel_5_O0501_reduced, ren_channel_6_O0603_reduced]]
#=== Main Update ====

print(raw.info['ch_names'])

look = folder_path.split('/')[-1]
try:
    index = np.argwhere(main_update[0] == look)[0][0]
    raw.rename_channels(mapping=main_update[1][index])
except:
    print('This file doesn`t need to be renamed at first')
    
print(raw.info['ch_names'])

#%%========== Area 51 (FILD) ========
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/FILD_Raw_data/Lucidity Found/FILD20_n1.edf'
raw = chetto_EEG.read_edf_file(folder_path)

#=== Pick Info ======
renamed_channels = {'EEG F3-A2':'F3', 'EEG F4-A1':'F4', 'EOG Left':'EOG-0','EOG Right':'EOG-1', 'EMG Chin-0':'EMG',\
                    'EEG C3-A2':'C3', 'EEG C4-A1':'C4', 'EEG O1-A2':'O1', 'EEG O2-A1':'O2'}
channel_types = {'EOG-0':'eog', 'EOG-1':'eog', 'F3':'eeg', 'F4':'eeg', 'EMG':'emg', 'C3':'eeg', 'C4':'eeg',\
                 'O1':'eeg', 'O2':'eeg', 'ECG':'ecg'}    
picked_channels = ['EOG-0', 'EOG-1', 'F3', 'F4', 'C3','C4','O1','O2', 'EMG','ECG']
#=== Pick Info ======

ten_twenty_montage = mne.channels.make_standard_montage('standard_1020') #this is chosen

#==== Montage Setup ======
raw.rename_channels(mapping=renamed_channels)
raw.pick_channels(ch_names=picked_channels)
raw.set_channel_types(mapping=channel_types)

raw.set_montage(ten_twenty_montage)
#==== Montage Setup ======

raw.plot_sensors(show_names=True)
print(raw.info)
#%%========== Area 51 (Erlacher) ========
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Final EDF Files/6_O0603!_reduced.edf'
count=1
raw = chetto_EEG.read_edf_file(folder_path)

#=== Main Update ==== 
# ren_channel_5_O0501_reduced = {'EOGH1':'EOGH-0', 'EOGH2':'EOGH-1'}
# ren_channel_6_O0603_reduced = {'C4-A1':'EEG_C4', 'C3-A2':'EEG_C3'}
# main_update = [np.array(['5_O0501!_reduced.edf','6_O0603!_reduced.edf']), [ren_channel_5_O0501_reduced, ren_channel_6_O0603_reduced]]
#=== Main Update ====

#=== Pick Info ======
renamed_channels_0 = {'EEG_C3':'C3', 'EEG_C4':'C4', 'EOGH-0':'EOG-0', 'EOGH-1':'EOG-1'}
channel_types_0 = {'EOG-0':'eog', 'EOG-1':'eog', 'C3':'eeg', 'C4':'eeg', 'EMG':'emg'}
picked_channels_0 = {'EOG-0', 'EOG-1', 'C3','C4', 'EMG','ECG'}
renamed_channels = list()
channel_types = list()
picked_channels = list()
for i in range(6): renamed_channels.append(renamed_channels_0), channel_types.append(channel_types_0), \
                   picked_channels.append(picked_channels_0)
                   
renamed_channels[1] = {'C3-A2':'C3', 'C4-A1':'C4'}
renamed_channels[4] = {'EOGH1':'EOG-0', 'EOGH2':'EOG-1'}
channel_types[1] =  {'C3':'eeg', 'C4':'eeg'}
picked_channels[1] = {'C3','C4', 'EMG','ECG'}

event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
#=== Pick Info ======

#===== Main Update ======
# if(main_update is not None):
#     look = folder_path.split('/')[-1]
#     try:
#         index = np.argwhere(main_update[0] == look)[0][0]
#         raw.rename_channels(mapping=main_update[1][index])
#         print('Initial channel rename applied!')
#     except:
#         print('This file doesn`t need to be renamed at first')
#===== Main Update ======

print(raw.info)

ten_twenty_montage = mne.channels.make_standard_montage('standard_1020') #this is chosen

#==== Montage Setup ======
raw.rename_channels(mapping=renamed_channels[count])
raw.pick_channels(ch_names=picked_channels[count])
raw.set_channel_types(mapping=channel_types[count])

raw.set_montage(ten_twenty_montage)
#==== Montage Setup ======

raw.plot_sensors(show_names=True)
print(raw.info)
#%%========== Area 51 (Sergio) ========
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Sergio/Dataset/EDF/2_Luane 2.edf'
raw = chetto_EEG.read_edf_file(folder_path)

#=== Pick Info ======
renamed_channels = {'EEG F7':'F7', 'EEG T3':'T3', 'EEG T5':'T5', 'EEG Fp1':'Fp1', 'EEG F3':'F3','EEG C3':'C3', 'EEG P3':'P3',\
                    'EEG O1':'O1', 'EEG F8':'F8', 'EEG T4':'T4', 'EEG T6':'T6', 'EEG Fp2': 'Fp2', 'EEG F4':'F4', 'EEG C4':'C4',\
                    'EEG P4':'P4', 'EEG O2':'O2', 'EEG Fz':'Fz','EEG Cz':'Cz', 'EEG Pz':'Pz', 'EEG Oz':'Oz', 'EEG A1':'A1',\
                    'EEG A2':'A2', 'Oc1':'EOG-0', 'Oc2':'EOG-1'}
channel_types = {'EOG-0':'eog', 'EOG-1':'eog', 'F7':'eeg','T3':'eeg','T5':'eeg', 'Fp1':'eeg', 'F3':'eeg',\
                 'C3':'eeg', 'P3':'eeg', 'O1':'eeg', 'F8':'eeg', 'T4':'eeg', 'T6':'eeg', 'Fp2':'eeg', 'F4':'eeg', 'C4':'eeg',\
                 'P4':'eeg', 'O2':'eeg', 'Fz':'eeg', 'Cz':'eeg', 'Pz':'eeg', 'Oz':'eeg', 'A1':'eeg', 'A2':'eeg', 'EMG':'emg','ECG':'ecg'}
picked_channels = list(channel_types.keys())
picked_channels.append('EMG')
picked_channels.append('ECG') #exclude 2 channels ('FOTO' and 'FreqC')
#=== Pick Info ======

ten_twenty_montage = mne.channels.make_standard_montage('standard_1020') #this is chosen

#==== Montage Setup ======
raw.rename_channels(mapping=renamed_channels)
raw.pick_channels(ch_names=picked_channels)
raw.set_channel_types(mapping=channel_types)

raw.set_montage(ten_twenty_montage)
#==== Montage Setup ======

raw.plot_sensors(show_names=True)
print(raw.info)
#%%========== Area 51 (Jarrod) ========
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/LD_EEG_Jarrot/EDF Files/Lucidity Found/AC P243_14-10-2019.edf'
raw = chetto_EEG.read_edf_file(folder_path)

#=== Pick Info ======
renamed_channels = {'EOG1:A2':'EOG-0','EOG2:A1':'EOG-1','F3:A2':'F3','F4:A1':'F4', 'C3:A2':'C3','C4:A1':'C4','O1:A2':'O1',\
                    'O2:A1':'O2', 'ECG 2':'ECG'}
channel_types = {'EOG-0':'eog','EOG-1':'eog','F3':'eeg','F4':'eeg','C3':'eeg','C4':'eeg','O1':'eeg','O2':'eeg', 'ECG':'ecg',\
                 'EMG':'emg'}  
picked_channels = ['EOG-0', 'EOG-1', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'EMG', 'ECG']
#=== Pick Info ======

ten_twenty_montage = mne.channels.make_standard_montage('standard_1020') #this is chosen

#==== Montage Setup ======
raw.drop_channels(ch_names=['EOG1', 'EOG2','F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'EMG+', 'EMG-', 'Light', 'Battery'])
raw.set_eeg_reference(ref_channels=['A1', 'A2'])
raw.rename_channels(mapping=renamed_channels)
raw.pick_channels(ch_names=picked_channels)
raw.set_channel_types(mapping=channel_types)

raw.set_montage(ten_twenty_montage)
#==== Montage Setup ======

raw.plot_sensors(show_names=True)
print(raw.info)
#%% ========== Area 51 (Lucireta) ========

from mne.channels import read_custom_montage
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Lucireta/General Data/3_AA0010AK_1-3+.edf'
fname_mon = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Everything About Total Data/Annotations&Labels&Information About Datasets/' + \
            'Stuff About Lucireta/DigitizingData/15-05-23 Loreta 94 2.2.elp'
# dig_montage = read_custom_montage(fname_mon, head_size=None, coord_frame=None)
#==== Read Custom Montage ======

#==== raw kit =====
# raw = mne.io.read_raw_kit(input_fname=folder_path, elp=fname_mon)
#==== raw kit =====

#===== read montage ======
raw = chetto_EEG.read_edf_file(folder_path)
# ch_names = raw.info['ch_names'][0:-4]
# ch_names.remove('E')
# ch_names.remove('EEG Mark1')
# ch_names.remove('EEG Mark2')

# montage = mne.channels.read_dig_polhemus_isotrak(fname=fname_mon, ch_names=ch_names)
# print(montage._get_ch_pos())
# dig_names = montage._get_dig_names()
# raw.set_montage(montage)
#===== read montage ======

# montage2 = mne.channels.make_standard_montage('standard_1020') #this is chosen

#==== 10-05 update ====
montage = mne.channels.make_standard_montage('standard_1005') #this is chosen
montage2 = mne.channels.make_standard_montage('standard_1005') #this is chosen
# montage.plot(kind='3d')
montage_chnames = montage.ch_names
dig_names = montage._get_dig_names()
ch_pos = montage._get_ch_pos()
ch_names = raw.info['ch_names']

ch_indexes = np.zeros(len(ch_names)) - 1
ch_indexes = ch_indexes.astype(int)
to_be_deleted = list()
for i in range(len(ch_names)):
    try:
        ch_indexes[i] = montage_chnames.index(ch_names[i])
    except:
        print(ch_names[i])
        to_be_deleted.append(i)
#==== 10-05 update ====

ch_indexes = np.delete(ch_indexes, to_be_deleted)
ch_names = np.delete(ch_names, to_be_deleted)
ch_names = ch_names.tolist()

#=== montage update ======
montage.dig = montage.dig[3:]
montage.ch_names = [montage.ch_names[index] for index in ch_indexes]
montage.dig = [montage.dig[index] for index in ch_indexes]
# montage.plot(kind='3d')

raw.drop_channels(ch_names=['FFC!h','E','SI5','SI3','SI6','SI4','IIz','Events/Markers','EEG Mark1','EEG Mark2'])
raw.set_eeg_reference(ref_channels=['TP7', 'TP9'])
raw.set_channel_types({'IO':'eog','ECG':'ecg','SM1':'emg','SM2':'emg','SM3':'emg'})
raw.set_montage(montage)
#=== montage update ======

raw.plot_sensors(show_names='True')

#%% ====== Event Creation ======= 
period_interval = lucidity_period_munich[1]
file_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Lucireta/General Data/1_AA00109T_1-4+.edf'
raw = chetto_EEG.read_edf_file(file_path)
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}

total_events = np.empty(shape=[0,len(event_id)])
for i in range(len(period_interval)):
        
    if(np.size(period_interval[i,0]) > 1): #if there are multiple intervals in a file
        for j in range(len(period_interval[i])):
            temp_event = mne.make_fixed_length_events(raw=raw, id=i, start=period_interval[i,j][0], \
                         stop=period_interval[i,j][1], duration=4, overlap=2)
            total_events = np.row_stack((total_events, temp_event))
    else:
        temp_event = mne.make_fixed_length_events(raw=raw, id=i, start=period_interval[i,0], stop=period_interval[i,1], \
                                                  duration=4, overlap=2)
        total_events = np.row_stack((total_events, temp_event))
     # ====== Event Creation ======= 
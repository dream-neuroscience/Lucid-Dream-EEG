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
os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes')

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
# lucidity_period_okniebeugen = np.array([[[16962, 17023.5],[17023.5, 17084.5],[17084.5, 17145]],\
#                                         [[22859, 22969],[22969, 23079.5],[23079.5, 23189.5]],\
#                                         [[28466, 28691],[28691, 28916],[28916, 29141]],\
#                                         [[31806, 31929],[31929, 32052],[32052, 32175]],\
#                                         [[22270, 22485],[22485, 22700],[22700, 22915]],\
#                                         [[28091, 28197.5], [28197.5, 28304],[28304,28410.5]],\
#                                         [[24603, 24696.5],[24696.5, 24839.5],[24839.5,24981]]])

# #pre-lucid-post (fild_10_n1, fild_14_n2, fild_17_n2, fild_20_n1, fild_21_n2, fild_4_n1, fild_8_n2)
# lucidity_period_fild = np.array([[[40140, 40396],[40397, 40587],[40587, 40620]],\
#                                 [[39257, 39263],[39264, 39271],[39272, 39279]],\
#                                 [[38313, 38383],[38383, 38463],[38463, 38543]],\
#                                 [[42876, 42969],[42969, 43062],[43062, 43155]],\
#                                 [[36033, 36151],[36151, 36344],[36344, 36537]],\
#                                 [[35820, 35919],[35919, 36024],[36024, 36054]],\
#                                 [[34020, 34041], [34041, 34168],[34170, 34200]]])

# #pre-lucid-post (0008001, 000201, 000401, Luane2)                                   
# lucidity_period_sergio = np.array([[[11010,11016],[11016,11069],[11069,11126]],\
#                                   [[7789.2, 7802.1],[7802.1, 7815],[7815, 7827.9]],\
#                                   [[27069.9, 27079.2],[27079.2, 27088.5],[27088.5, 27097.8]],\
#                                   [[92.2 ,99.9],[99.9, 107.6],[107.6, 115.3]]]) 

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

#rem-lucid-wake  (203,301,302,304_2,303,501) 406 excluded
lucidity_period_okniebeugen = np.array([[[16770, 16831],[17023.5, 17084.5],[51, 112]],\
                                        [[22800, 22910.5],[22969, 23079.5],[1051, 1161.5]],\
                                        [[28410, 28635],[28691, 28916],[2058, 2283]],\
                                        [[31806, 31909],[31929, 32052],[1045, 1168]],\
                                        [[22320, 22465],[22485, 22700],[618, 833]],\
                                        [[27150, 27256.5],[28212.5, 28251.6],[1108, 1215]]])
                                        #[[30457, 30491],[30916.5, 30950],[31880.85, 32000]]]) this is 406
lengths = lucidity_period_okniebeugen[:,1,1] - lucidity_period_okniebeugen[:,1,0]

#rem-lucid-wake (0008001, 000401, Luane2)                                   
# lucidity_period_sergio = np.array([[[3912,3965],[11016,11069],[10815,10868]],\
#                                   [[24509, 24518.3],[27079.2, 27088.5],[523, 532.3]],\
#                                   [[7230 ,7237],[99.9, 107.6],[12191, 12198.7]]]) 
lucidity_period_sergio = np.array([[[24509, 24518.3],[27079.2, 27088.5],[523, 532.3]],\
                                  [[7230 ,7237],[99.9, 107.6],[12191, 12198.7]]]) 
lengths = lucidity_period_sergio[:,1,1] - lucidity_period_sergio[:,1,0]

#rem-lucid-wake (AC_P243, VR_P261_OLI_1, VR_P261_OLI_2, VR_P277)
# lucidity_period_jarrod = np.array([[[19206, 19226],[19372, 19392.5],[24732, 24752]],\
#                                   [[12453, 12559],[[13463.5, 13569],[18302, 18352]],[13596, 13674]],\
#                                   [[35976, 36016],[33185, 33224],[5673, 5712]],\
#                                   [[27513, 27801],[[21126, 21159],[40077, 40182]],[7506, 7722]]], dtype=object)
lucidity_period_jarrod = np.array([[[19206, 19226],[19372, 19392.5],[24732, 24752]],\
                                  [[12453, 12559],[18302, 18352],[13596, 13674]],\
                                  [[35976, 36016],[33185, 33224],[5673, 5712]],\
                                  [[27513, 27801],[40077, 40182],[7506, 7722]]])    

# s1, s6, s7, s8, s9, s11, s12, s13, s14, s17
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
lucidity_period_munic_wake = np.array([[[4,17],[1895,1907],[25,35],[1475,1485]]])
lucidity_period_munic_wake = np.reshape(lucidity_period_munic_wake, (4,1,2))

baseline = (0, 0.2)

#======= Single EDFs =========
#ZUG_05_Schmitt_2
lucidity_period_zugschmitt = np.array([[34380, 34500],[43843.5, 44011],[44124, 44140]])
# lucidity_period_zugschmitt = np.resize(lucidity_period_zugschmitt, (1,3,2))

#O0603
lucidity_period_O0603 = np.array([[23910, 24000],[24696.5, 24839.5],[909, 1052]])
# lucidity_period_O0603 = np.resize(lucidity_period_O0603, (1,3,2))
#======= Single EDFs =========

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

# SSPs, CSDs = [False, True, False, True], [False, False, True, True]
CSDs = [False, True]
picks = ['eeg', 'csd']

paths = ['/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD No (Unit Factor Normalized)',\
         '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes (Unit Factor Normalized)']
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes']
for i in range(2):
    os.chdir(paths[i])
    ICA, SSP, CSD = True, False, CSDs[i]
    
    fild_epochs, fild_events, fild_listepochs, _ = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, renamed_channels=renamed_channels, \
                               channel_types=channel_types, picks=picks[i], picked_channels=picked_channels, event_id=event_id, f_min=0.1, f_max=48,\
                               f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, baseline=None,\
                               ICA=ICA, SSP=SSP, CSD=CSD, resample=100, preload=True)
        
    pickle.dump([fild_epochs, fild_events, fild_listepochs], open('fild_epochs_events_nobaseline','wb'))
#%% ====== Epoch Concatenation O_kniebeugen ======
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Final EDF Files'

#=== Pick Info ======
renamed_channels_0 = {'EEG_C3':'C3', 'EEG_C4':'C4', 'EOGH-0':'EOG-0', 'EOGH-1':'EOG-1', 'EKG':'ECG'}
channel_types_0 = {'EOG-0':'eog', 'EOG-1':'eog', 'C3':'eeg', 'C4':'eeg', 'EMG':'emg', 'ECG':'ecg'}
picked_channels_0 = {'EOG-0', 'EOG-1', 'C3','C4', 'EMG','ECG'}
renamed_channels = list()
channel_types = list()
picked_channels = list()
for i in range(8): renamed_channels.append(renamed_channels_0), channel_types.append(channel_types_0), \
                   picked_channels.append(picked_channels_0)
                   
# renamed_channels[6] = {'C3-A2':'C3', 'C4-A1':'C4'}
renamed_channels[5] = {'EOGH1':'EOG-0', 'EOGH2':'EOG-1', 'EKG':'ECG'}
# renamed_channels[6] = {'C4-A1':'C4', 'C3-A2':'C3', 'EKG':'ECG'}
renamed_channels[6] = {'EOGH1':'EOG-0', 'EOGH2':'EOG-1', 'EKG':'ECG'}
# channel_types[6] = {'C3':'eeg', 'C4':'eeg', 'ECG':'ecg'}
# picked_channels[6] = {'C3','C4', 'ECG'}

period_interval = lucidity_period_okniebeugen
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
#=== Pick Info ======

picks = ['eeg']
paths = ['/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA No, SSP No, CSD No (Unit Factor Normalized)']
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes']

# for i in range(2):
i=0
os.chdir(paths[i])
ICA, SSP, CSD = False, False, False
   
o_kniebeugen_epochs, o_kniebeugen_events, o_kniebeugen_listepoches,_ = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, \
renamed_channels=renamed_channels, channel_types=channel_types, picks=['eeg','eog'], picked_channels=picked_channels, \
event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
baseline=None, ICA=ICA, SSP=SSP, CSD=CSD, input_as_list=True, resample=100, preload=True)
    
pickle.dump([o_kniebeugen_epochs, o_kniebeugen_events, o_kniebeugen_listepoches], open('o_kniebeugen_epochs_events_nobaseline','wb'))
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

CSDs = [False, True]
picks = ['eeg', 'csd']
paths = ['/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD No (Unit Factor Normalized)',\
         '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes (Unit Factor Normalized)']
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes']
for i in range(2):
    os.chdir(paths[i])
    ICA, SSP, CSD = True, False, CSDs[i]

    sergio_epochs, sergio_events, sergio_listepoches, _ = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, \
    renamed_channels=renamed_channels, channel_types=channel_types, picks=picks[i], picked_channels=picked_channels, \
    event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
    baseline=None, ICA=ICA, SSP=SSP, CSD=CSD, resample=100, preload=True)
        
    pickle.dump([sergio_epochs, sergio_events, sergio_listepoches], open('sergio_epochs_events_nobaseline','wb'))
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

CSDs = [False, True]
picks = ['eeg', 'csd']
paths = ['/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD No (Unit Factor Normalized)',\
         '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes (Unit Factor Normalized)']
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes']
for i in range(2):
    os.chdir(paths[i])
    ICA, SSP, CSD = True, False, CSDs[i]

    jarrot_epochs, jarrot_events, jarrot_listepoches, _ = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, \
    renamed_channels=renamed_channels, channel_types=channel_types, picks=picks[i], picked_channels=picked_channels, \
    event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
    baseline=None, ICA=ICA, SSP=SSP, CSD=CSD, drop_channels=drop_channels, eeg_reference=eeg_reference, resample=100, \
    preload=True)
        
    pickle.dump([jarrot_epochs, jarrot_events, jarrot_listepoches], open('jarrot_epochs_events_nobaseline','wb'))
#%% ====== Epoch Concatenation Lucireta ======
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Lucireta/General Data'
folder_path_wake = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Lucireta/Wake Data'

#=== Pick Info ======
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
period_interval = lucidity_period_munich

common_bad_channels = [['E','ECG','EEG Mark1','EEG Mark2','Events/Markers','F1','F10','F8','F9','FFC3h','FFC5h','I2','IIz','IO','O2','P2','PO10','PO8','SI3','SI4','SI5','SI6','SM1','SM2','SM3','TP9'],\
                        ['E','ECG','EEG Mark1','EEG Mark2','Events/Markers','IIz','IO','P2','SI3','SI4','SI5','SI6','SM1','SM2','SM3'],\
                        ['E','ECG','EEG Mark1','EEG Mark2','Events/Markers','IIz','IO','SI3','SI4','SI5','SI6','SM1','SM2','SM3','TP10'],\
                        ['AF3','E','ECG','EEG Mark1','EEG Mark2','Events/Markers','F3','FC2','IIz','IO','SI3','SI4','SI5','SI6','SM1','SM2','SM3'],\
                        ['F2','AF3','AFF1h','AFF5h','AFp1','AFz','E','ECG','EEG Mark1','EEG Mark2','Events/Markers','F3','F5','FFC5h','IIz','IO','SI3','SI4','SI5','SI6','SM1','SM2','SM3'],\
                        ['AF3','AFF1h','AFF5h','E','ECG','EEG Mark1','EEG Mark2','Events/Markers','IIz','IO','SI3','SI4','SI5','SI6','SM1','SM2','SM3'],\
                        ['AF3','E','ECG','EEG Mark1','EEG Mark2','Events/Markers','IIz','IO','SI3','SI4','SI5','SI6','SM1','SM2','SM3','T8'],\
                        ['AFF6h','E','ECG','EEG Mark1','EEG Mark2','Events/Markers','F1','FFC4h','IIz','IO','Oz','SI3','SI4','SI5','SI6','SM1','SM2','SM3']]

temp = list()                   
for i in range(len(common_bad_channels)):
    temp = list(set(temp) or set(common_bad_channels[i]))
temp = sorted(temp)
temp.pop(0)
temp.pop(0)
temp.pop(11)
temp.pop(-1)
drop_channels = temp

drop_channels = ['F1','F10','F8','F9','FFC3h', 'FFC5h', 'I2', 'O2', 'P2', 'PO10', 'PO8'] #final decision
#=== Pick Info ======

CSDs = [False, True]
picks = ['eeg', 'csd']
paths = ['/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD No (Unit Factor Normalized)',\
         '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes (Unit Factor Normalized)']
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes']
for i in range(2):
    os.chdir(paths[i])
    ICA, SSP, CSD = True, False, CSDs[i]

    lucireta_epochs, lucireta_events, lucireta_listepoches, _ = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path, \
    picks='csd', event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, \
    tmax = None, baseline=None, ICA=ICA, SSP=SSP, CSD=CSD, Lucireta=True, resample=100)
    
    pickle.dump([lucireta_epochs, lucireta_events, lucireta_listepoches], open('lucireta_epochs_events_nobaseline','wb'))
    
    #Just awakes
    event_id = {'Wake': 2}
    lucireta_epochs_wake, lucireta_events_wake, lucireta_wake_listepochs, _ = chetto_EEG.eeg_epoching_pipeline_of_given_folder(folder_path=folder_path_wake, \
    picks='csd', event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=lucidity_period_munic_wake, duration=4, \
    overlap=2, tmin=0, tmax = None, baseline=None, ICA=ICA, SSP=SSP, CSD=CSD, Lucireta=True, resample=100, single_event_id=2)
    
    pickle.dump([lucireta_epochs_wake, lucireta_wake_listepochs], open('lucireta_epochs_events_wake_nobaseline','wb'))

    #===== Lucireta Event & Epoch Merge =========
    # os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes')
    lucireta_epochs, lucireta_events, lucireta_epochlist = pickle.load(open('lucireta_epochs_events_nobaseline','rb'))
    lucireta_epochs_wake, lucireta_events_wake, _ = pickle.load(open('lucireta_epochs_events_wake_nobaseline','rb'))
    
    # #indexes to be changed
    # #wake 0-5 --> 19-24
    # #wake 5-10 --> 99-104 (99-103 + 1)
    # #wake 10-14 --> 124-128 (124-133 - 5)
    # #wake 14-18 --> 136-140
    
    # #change events (luckily same amount)
    # lucireta_events[34:39] = lucireta_events_wake[5:10]
    # lucireta_events[63:67] = lucireta_events_wake[14:18]
    
    lucireta_events = lucireta_epochs.events
    lucireta_epochs._data[19:24] = lucireta_epochs_wake._data[0:5]
    lucireta_epochs._data[99:103] = lucireta_epochs_wake._data[5:9]
    lucireta_epochs._data[123:127] = lucireta_epochs_wake._data[10:14]
    lucireta_epochs._data[140:144] = lucireta_epochs_wake._data[14:18]
    
    lucireta_epochlist[0]._data[0:5] = lucireta_epochs_wake._data[0:5]
    lucireta_epochlist[5]._data[0:4] = lucireta_epochs_wake._data[5:9]
    lucireta_epochlist[6]._data[0:4] = lucireta_epochs_wake._data[10:14]
    lucireta_epochlist[7]._data[0:4] = lucireta_epochs_wake._data[14:18]
    
    pickle.dump([lucireta_epochs, lucireta_events], open('lucireta_epochs_events_final_nobaseline','wb'))
#%% ======== Single EDF O0603.edf =================

file_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Final EDF Files/Other EDF File/6_O0603!_reduced.edf'

#===== Pick Info ========
renamed_channels = {'C4-A1':'C4', 'C3-A2':'C3', 'EKG':'ECG', 'F4-A1':'F4', 'F3-A2':'F3'}
channel_types = {'C3':'eeg', 'C4':'eeg', 'F4':'eeg', 'F3':'eeg', 'ECG':'ecg'}
picked_channels = {'C3','C4','F3','F4','ECG'}
#===== Pick Info ========

period_interval = lucidity_period_O0603
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
#=== Pick Info ======

picks = ['eeg']
paths = ['/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA No, SSP No, CSD No (Unit Factor Normalized)']
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes']

# for i in range(2):
i=0
os.chdir(paths[i])
ICA, SSP, CSD = False, False, False

raw = chetto_EEG.eeg_read_sensorupdate_file(file_path=file_path, renamed_channels=renamed_channels,\
                                      channel_types=channel_types, picks='eeg', drop_channels=None, \
                                      picked_channels=picked_channels, eeg_reference=None, \
                                      Lucireta=False, input_as_list=False,\
                                      preload=True)
raw = chetto_EEG.eeg_preprocess(raw=raw, f_notch=None, f_min=0.1, f_max=48, ICA=ICA, SSP=SSP, CSD=CSD, 
                                resample=100, count=0)
# raw = chetto_EEG.eeg_preprocess_file(file_path=file_path, renamed_channels=renamed_channels,\
#                                      channel_types=channel_types, picks='eeg', drop_channels=None, \
#                                      picked_channels=picked_channels, eeg_reference=None, \
#                                      f_min=0.1, f_max=48, f_notch=49, ICA=ICA, SSP=SSP, \
#                                      CSD=CSD, Lucireta=False, count=0, input_as_list=False,\
#                                      resample=100, preload=True) 
# ===== Epoch / Event Creation ======
temp_epochs, temp_events = chetto_EEG.event_epoch_creation(raw=raw, event_id=event_id, period_interval=period_interval, \
                                                     duration=4, picks='eeg', overlap=2, tmin=0, tmax=None, \
                                                     baseline=None)
    
pickle.dump([temp_epochs, temp_events], open('O0603_epochs_events_nobaseline','wb'))    
# ===== Epoch / Event Creation ======
#%% ======== Single EDF ZUG_05_Schmitt_2 =================

file_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Zug Schmitt/ZUG_05_Schmitt_2_filtered.edf'

#===== Pick Info ========
renamed_channels = {'C4 - A1':'C4', 'C3 - A2':'C3', 'LOC':'EOG-0', 'ROC':'EOG-1',\
                    'CHIN1':'EMG-0', 'CHIN2':'EMG-1'}
channel_types = {'C3':'eeg', 'C4':'eeg', 'EOG-0':'eog', 'EOG-1':'eog', 'EMG-0':'emg', 'EMG-1':'emg'}
picked_channels = {'C3','C4', 'EOG-0', 'EOG-1', 'EMG-0', 'EMG-1'}
#===== Pick Info ========

period_interval = lucidity_period_zugschmitt
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
#=== Pick Info ======

picks = ['eeg']
paths = ['/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA No, SSP No, CSD No (Unit Factor Normalized)']
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes',\
         # '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes']

# for i in range(2):
i=0
os.chdir(paths[i])
ICA, SSP, CSD = False, False, False

raw = chetto_EEG.eeg_read_sensorupdate_file(file_path=file_path, renamed_channels=renamed_channels,\
                                      channel_types=channel_types, picks='eeg', drop_channels=None, \
                                      picked_channels=picked_channels, eeg_reference=None, \
                                      Lucireta=False, input_as_list=False,\
                                      preload=True)
raw = chetto_EEG.eeg_preprocess(raw=raw, f_notch=None, f_min=0.1, f_max=48, ICA=ICA, SSP=SSP, CSD=CSD, 
                                resample=100, count=0)
# raw = chetto_EEG.eeg_preprocess_file(file_path=file_path, renamed_channels=renamed_channels,\
#                                      channel_types=channel_types, picks='eeg', drop_channels=None, \
#                                      picked_channels=picked_channels, eeg_reference=None, \
#                                      f_min=0.1, f_max=48, f_notch=49, ICA=ICA, SSP=SSP, \
#                                      CSD=CSD, Lucireta=False, count=0, input_as_list=False,\
#                                      resample=100, preload=True) 
# ===== Epoch / Event Creation ======
temp_epochs, temp_events = chetto_EEG.event_epoch_creation(raw=raw, event_id=event_id, period_interval=period_interval, \
                                                     duration=4, picks='eeg', overlap=2, tmin=0, tmax=None, \
                                                     baseline=None)
        
pickle.dump([temp_epochs, temp_events], open('zug05schmitt2_epochs_events_nobaseline','wb'))    
# ===== Epoch / Event Creation ======
    
#%% =========== Spectrogram Analysis of Neighbourhood Data (FILD) ==============
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

ICA, SSP, CSD = True, False, False

raw_data, fake_epochs, new_periods = chetto_EEG.eeg_neighbourhoodinterval_of_given_folder(folder_path=folder_path, renamed_channels=renamed_channels, \
                           channel_types=channel_types, picks='eeg', picked_channels=picked_channels, event_id=event_id, f_min=0.1, f_max=48,\
                           f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, baseline=(0, 0.2),\
                           ICA=ICA, SSP=SSP, CSD=CSD, resample=100, preload=True)

os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval')
pickle.dump([raw_data, fake_epochs, new_periods], open('FILD_lucidneighbour_interval_icayessspnocsdno','wb'))    
#%% =========== Spectrogram Analysis of Neighbourhood Data (Erlacher) ==============
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Final EDF Files'

#=== Pick Info ======
renamed_channels_0 = {'EEG_C3':'C3', 'EEG_C4':'C4', 'EOGH-0':'EOG-0', 'EOGH-1':'EOG-1', 'EKG':'ECG'}
channel_types_0 = {'EOG-0':'eog', 'EOG-1':'eog', 'C3':'eeg', 'C4':'eeg', 'EMG':'emg', 'ECG':'ecg'}
picked_channels_0 = {'EOG-0', 'EOG-1', 'C3','C4', 'EMG','ECG'}
renamed_channels = list()
channel_types = list()
picked_channels = list()
for i in range(8): renamed_channels.append(renamed_channels_0), channel_types.append(channel_types_0), \
                   picked_channels.append(picked_channels_0)

#==== Changes =====                   
renamed_channels[5] = {'EOGH1':'EOG-0', 'EOGH2':'EOG-1', 'EKG':'ECG'}

renamed_channels[6] = {'C4-A1':'C4', 'C3-A2':'C3', 'EKG':'ECG', 'F4-A1':'F4', 'F3-A2':'F3'}
channel_types[6] = {'C3':'eeg', 'C4':'eeg', 'F4':'eeg', 'F3':'eeg', 'ECG':'ecg'}
picked_channels[6] = {'C3','C4','F3','F4','ECG'}

renamed_channels[7] = {'C4 - A1':'C4', 'C3 - A2':'C3', 'LOC':'EOG-0', 'ROC':'EOG-1',\
                    'CHIN1':'EMG-0', 'CHIN2':'EMG-1'}
channel_types[7] = {'C3':'eeg', 'C4':'eeg', 'EOG-0':'eog', 'EOG-1':'eog', 'EMG-0':'emg', 'EMG-1':'emg'}
picked_channels[7] = {'C3','C4', 'EOG-0', 'EOG-1', 'EMG-0', 'EMG-1'}
#==== Changes =====     

lucidity_period_zugschmitt_copy = np.resize(lucidity_period_zugschmitt, (1,3,2))
lucidity_period_O0603_copy = np.resize(lucidity_period_O0603, (1,3,2))
lucidity_period_okniebeugen = np.row_stack((lucidity_period_okniebeugen, lucidity_period_O0603_copy))
lucidity_period_okniebeugen = np.row_stack((lucidity_period_okniebeugen, lucidity_period_zugschmitt_copy))

period_interval = lucidity_period_okniebeugen
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
#=== Pick Info ======

ICA, SSP, CSD = True, False, False

raw_data, fake_epochs, new_periods = chetto_EEG.eeg_neighbourhoodinterval_of_given_folder(folder_path=folder_path, renamed_channels=renamed_channels, \
                           channel_types=channel_types, picks='eeg', picked_channels=picked_channels, event_id=event_id, f_min=0.1, f_max=48,\
                           f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, baseline=(0, 0.2),\
                           ICA=ICA, SSP=SSP, CSD=CSD, resample=100, preload=True, input_as_list=True)

os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval')
pickle.dump([raw_data, fake_epochs, new_periods], open('Erlacher_lucidneighbour_interval','wb'))       
# lucid_begin = new_periods[1,0]
# lucid_finish = new_periods[1,1]
#%% ============= Spectrogram Analysis of Neighbourhood Data (Sergio)
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

ICA, SSP, CSD = True, False, False

raw_data, fake_epochs, new_periods = chetto_EEG.eeg_neighbourhoodinterval_of_given_folder(folder_path=folder_path, \
renamed_channels=renamed_channels, channel_types=channel_types, picks='eeg', picked_channels=picked_channels, \
event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
baseline=(0, 0.2), ICA=ICA, SSP=SSP, CSD=CSD, resample=100)
    
os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval')
pickle.dump([raw_data, fake_epochs, new_periods], open('Sergio_lucidneighbour_interval','wb')) 
#%% ============= Spectrogram Analysis of Neighbourhood Data (Jarrot)
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

ICA, SSP, CSD = True, True, False

raw_data, fake_epochs, new_periods = chetto_EEG.eeg_neighbourhoodinterval_of_given_folder(folder_path=folder_path, \
renamed_channels=renamed_channels, channel_types=channel_types, picks='eeg', picked_channels=picked_channels, \
event_id=event_id, f_min=0.1, f_max=35, f_notch=None, period_interval=period_interval, duration=4, overlap=2, tmin=0, tmax = None, \
baseline=(0, 0.2), ICA=ICA, SSP=SSP, CSD=CSD, drop_channels=drop_channels, eeg_reference=eeg_reference, resample=100, preload=True)
    
os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval')
pickle.dump([raw_data, fake_epochs, new_periods], open('Jarrot_lucidneighbour_interval','wb'))    
#%% ============= Spectrogram Analysis of Neighbourhood Data (Lucireta)
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Lucireta/General Data'

#=== Pick Info ======
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}
period_interval = lucidity_period_munich

common_bad_channels = [['E','ECG','EEG Mark1','EEG Mark2','Events/Markers','F1','F10','F8','F9','FFC3h','FFC5h','I2','IIz','IO','O2','P2','PO10','PO8','SI3','SI4','SI5','SI6','SM1','SM2','SM3','TP9'],\
                        ['E','ECG','EEG Mark1','EEG Mark2','Events/Markers','IIz','IO','P2','SI3','SI4','SI5','SI6','SM1','SM2','SM3'],\
                        ['E','ECG','EEG Mark1','EEG Mark2','Events/Markers','IIz','IO','SI3','SI4','SI5','SI6','SM1','SM2','SM3','TP10'],\
                        ['AF3','E','ECG','EEG Mark1','EEG Mark2','Events/Markers','F3','FC2','IIz','IO','SI3','SI4','SI5','SI6','SM1','SM2','SM3'],\
                        ['F2','AF3','AFF1h','AFF5h','AFp1','AFz','E','ECG','EEG Mark1','EEG Mark2','Events/Markers','F3','F5','FFC5h','IIz','IO','SI3','SI4','SI5','SI6','SM1','SM2','SM3'],\
                        ['AF3','AFF1h','AFF5h','E','ECG','EEG Mark1','EEG Mark2','Events/Markers','IIz','IO','SI3','SI4','SI5','SI6','SM1','SM2','SM3'],\
                        ['AF3','E','ECG','EEG Mark1','EEG Mark2','Events/Markers','IIz','IO','SI3','SI4','SI5','SI6','SM1','SM2','SM3','T8'],\
                        ['AFF6h','E','ECG','EEG Mark1','EEG Mark2','Events/Markers','F1','FFC4h','IIz','IO','Oz','SI3','SI4','SI5','SI6','SM1','SM2','SM3']]

temp = list()                   
for i in range(len(common_bad_channels)):
    temp = list(set(temp) or set(common_bad_channels[i]))
temp = sorted(temp)
temp.pop(0)
temp.pop(0)
temp.pop(11)
temp.pop(-1)
drop_channels = temp

drop_channels = ['F1','F10','F8','F9','FFC3h', 'FFC5h', 'I2', 'O2', 'P2', 'PO10', 'PO8'] #final decision
#=== Pick Info ======

ICA, SSP, CSD = True, False, True

raw_data, fake_epochs, new_periods = chetto_EEG.eeg_neighbourhoodinterval_of_given_folder(folder_path=folder_path, \
picks='csd', event_id=event_id, f_min=0.1, f_max=48, f_notch=49, period_interval=period_interval, duration=4, overlap=2, tmin=0, \
tmax = None, baseline=(0, 0.2), ICA=ICA, SSP=SSP, CSD=CSD, Lucireta=True, resample=100)
    
os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval')
pickle.dump([raw_data, fake_epochs, new_periods], open('Lucireta_lucidneighbour_interval','wb'))    

#%% ============= Factor Analysis ===================
#FILD
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/FILD_Raw_data/Lucidity Found'
# FILD_statistics = chetto_EEG.factor_statistics_of_dataset(folder_path)
FILD_unitinfo = chetto_EEG.edf_deep_info_retrieval_of_dataset(folder_path)
#Erlacher
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Final EDF Files'
# Erlacher_statistics = chetto_EEG.factor_statistics_of_dataset(folder_path)
Erlacher_unitinfo = chetto_EEG.edf_deep_info_retrieval_of_dataset(folder_path)

#Sergio
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Sergio/Dataset/EDF'
# Sergio_statistics = chetto_EEG.factor_statistics_of_dataset(folder_path)
Sergio_unitinfo = chetto_EEG.edf_deep_info_retrieval_of_dataset(folder_path)

#Jarrod
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/LD_EEG_Jarrot/EDF Files/Lucidity Found'
# Jarrod_statistics = chetto_EEG.factor_statistics_of_dataset(folder_path)
Jarrod_unitinfo = chetto_EEG.edf_deep_info_retrieval_of_dataset(folder_path)

#Lucireta
folder_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Lucireta/General Data'
# Lucireta_statistics = chetto_EEG.factor_statistics_of_dataset(folder_path)
Lucireta_unitinfo = chetto_EEG.edf_deep_info_retrieval_of_dataset(folder_path)

#Erlacher_O0603
file_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Final EDF Files/6_O0603!_reduced.edf'
O0603_statistics = chetto_EEG.factor_statistics_of_file(file_path)

#Zug_schmitt2
file_path = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Zug Schmitt/ZUG_05_Schmitt_2_filtered.edf'
Zugschmitt2_statistics = chetto_EEG.factor_statistics_of_file(file_path)
#
#%% =============== FILD Load Data for Spectrogram Analysis ============
# from mne.time_frequency import tfr_multitaper

# os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval')
# FILD_raw_data, FILD_epoch, FILD_periods = pickle.load(open('FILD_lucidneighbour_interval','rb'))

# temp_epoch = FILD_epoch[0]

# #Multi-taper Parameters
# freqs = np.arange(2, 48, 0.1)  # frequencies from 2-35Hz
# n_cycles = freqs  # use constant t/f resolution
# vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
# n_cycles = freqs * 2  # use constant t/f resolution

# power = tfr_multitaper(temp_epoch, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, time_bandwidth = 8.0, decim=2,\
#                        average=True)
# baseline_period = 5 #second
# beginner = power.times[0]
# finish = power.times[-1]
# states = lucidity_period_fild[0][1]
# difference = states[1] - states[0]

# power.apply_baseline([beginner, beginner+20], mode='mean')

# avg_power = np.mean(power._data, 0)
# power._data = np.expand_dims(avg_power, axis=0) #expand dimension from axis=0 [1,x,y]

# power._data = np.absolute(power._data) ** (1/2.)
# power._data = 10 * np.log10(power._data)
# power._data = power._data[power._data < 0] = 0
#%% =============== Erlacher Load Data for Spectrogram Analysis ============
# from mne.time_frequency import tfr_multitaper

# os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval')
# _, erlacher_epoch, _ = pickle.load(open('Erlacher_lucidneighbour_interval','rb'))

# temp_epoch = erlacher_epoch[0]

# #Multi-taper Parameters
# freqs = np.arange(2, 48, 0.1)  # frequencies from 2-35Hz
# n_cycles = freqs  # use constant t/f resolution
# vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
# n_cycles = freqs * 2  # use constant t/f resolution

# power = tfr_multitaper(temp_epoch, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, time_bandwidth = 8.0, decim=2,\
#                        average=True)
# baseline_period = 5 #second
# beginner = power.times[0]
# finish = power.times[-1]
# states = lucidity_period_okniebeugen[0][1]
# difference = states[1] - states[0]
# text_location_difference = (difference * 3) * 0.033

# power.apply_baseline([beginner, beginner+20], mode='mean')

# avg_power = np.mean(power._data, 0)
# power._data = np.expand_dims(avg_power, axis=0) #expand dimension from axis=0 [1,x,y]

# power._data = np.absolute(power._data) ** (1/2.)
# power._data = 10 * np.log10(power._data)
# power._data = power._data[power._data < 0] = 0
#%%======== FILD Spectrogram Plot =========
# freq_thesholds = np.array([4,8,12,30])
# vmin=np.min(power._data)
# vmax=np.max(power._data)

# fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 1]})
# power.plot([0], vmin=vmin, vmax=vmax, axes=ax[0], colorbar=False, show=False)

# ax[0].axvline(0, linewidth=1, color="black", linestyle=":")  # event
# ax[0].legend(fontsize=15)
# fig.colorbar(ax[0].images[-1], cax=ax[-1])

# ax[0].text(beginner-text_location_difference,3, 'Delta', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
# ax[0].text(beginner-text_location_difference,6, 'Theta', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
# ax[0].text(beginner-text_location_difference,10, 'Alpha', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
# ax[0].text(beginner-text_location_difference,18, 'Beta', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
# ax[0].text(beginner-text_location_difference,37, 'Gamma', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')

# ax[0].text(states[0] - difference/2, 49, 'REM', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
# ax[0].text(states[0] + difference/2, 49, 'Lucid', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')
# ax[0].text(states[1] + difference/2, 49, 'Wake', horizontalalignment='center', verticalalignment='center', fontsize=15, color='Black')

# ax[0].vlines(x=states, ymin=0, ymax=48, colors='purple', ls='--', lw=3, label='States')
# ax[0].hlines(y=freq_thesholds, xmin=beginner, xmax=finish, colors='black', ls='--', lw=3, label='Brain waves')

# # ax[0].set_xlim(0, 100)
# plt.title('Color Spectrum', fontsize=13)
# plt.suptitle("Lucid Dream with Neighbourhood Spectrogram", fontsize=20)
# fig.show()

# mng = plt.get_current_fig_manager()
# mng.window.showMaximized()
# saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Spectrograms/Lucid & REM & Wake/Erlacher'
# plt.savefig(saving_directory + '/' + 'O_Knienebeugen Spectrogram of REM & Lucid & Wake' + '.jpeg', pad_inches=1, bbox_inches='tight', dpi=200)
# print('Figure has saved successfully!')
# plt.close()
#%% ======== Area 51 ============
raw = chetto_EEG.read_edf_file('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/2_Christian_Tausch/Zug Schmitt/ZUG_05_Schmitt_2_filtered.edf',
                               preload=True)
raw = chetto_EEG.read_edf_file('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/FILD_Raw_data/Lucidity Found/FILD8_n2.edf',
                               preload=False)
raw = chetto_EEG.read_edf_file('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Total Data/Lucireta/General Data/0_AA00108P_1-3+.edf',
                               preload=False)
raw._raw_extras[0]['physical_max'][0] * raw._raw_extras[0]['units'][0]
# raw.annotations
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
import extremeMachineLearning as chetto_ML
import pickle
# from mne.preprocessing import ICA
chetto_EEG = chetto_EEG.extremeEEGSignalAnalyzer()
chetto_ML = chetto_ML.extremeMachineLearning()
# %matplotlib qt
os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes')

#%% ===== Load Events & Epochs No ICA, No SSP, No CSP ============
fild_epochs, fild_events = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/No ICA, CSD, SSP/fild_epochs_events','rb'))
erlacher_epochs, erlacher_events = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/No ICA, CSD, SSP/o_kniebeugen_epochs_events','rb'))
jarrot_epochs, jarrot_events = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/No ICA, CSD, SSP/jarrot_epochs_events','rb'))
sergio_epochs, sergio_events = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/No ICA, CSD, SSP/sergio_epochs_events','rb'))
lucireta_epochs, lucireta_events = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/No ICA, CSD, SSP/lucireta_epochs_events_final','rb'))
#%% ===== Load Events & Epochs Yes ICA, Yes SSP, No CSP ============
fild_epochs_icassp, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No/fild_epochs_events','rb'))
erlacher_epochs_icassp, _ , erlacher_epochlist_icassp = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No/o_kniebeugen_epochs_events','rb'))
jarrot_epochs_icassp, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No/jarrot_epochs_events','rb'))
sergio_epochs_icassp, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No/sergio_epochs_events','rb'))
lucireta_epochs_icassp, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD No/lucireta_epochs_events_final','rb'))
#%% ===== Load Events & Epochs Yes ICA, Yes SSP, Yes CSP ============
fild_epochs_icasspcsd, fild_events, fild_epochslist_icasspcsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes/fild_epochs_events','rb'))
jarrot_epochs_icasspcsd, jarrot_events, jarrot_epochlist_icasspcsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes/jarrot_epochs_events','rb'))
sergio_epochs_icasspcsd, sergio_events, sergio_epochlist_icasspcsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes/sergio_epochs_events','rb'))
lucireta_epochs_icasspcsd_rereferenced, lucireta_events = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes/lucireta_epochs_events_final_rereferenced','rb'))
lucireta_epochs_icasspcsd, lucireta_events, lucireta_epochlist_icasspcsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes/lucireta_epochs_events_final','rb'))
#%% ==== Periodograms ====

#====== FILD ==========
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/FILD/No ICA, No SSP, No CSD'

periodogram_avg = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=fild_epochs, events=fild_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='FILD, Lucid vs. REM vs. Wake', saving_directory=saving_directory)

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/FILD/No ICA, Yes SSP, No CSD'

periodogram_avg_icassp = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=fild_epochs_icassp, events=fild_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='FILD, Lucid vs. REM vs. Wake', saving_directory=saving_directory)
    
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/FILD/No ICA, Yes SSP, Yes CSD'

periodogram_avg_icasspcsd = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=fild_epochs_icasspcsd, events=fild_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='FILD, Lucid vs. REM vs. Wake', saving_directory=saving_directory)
#====== FILD ==========

#%%====== Erlacher ==========
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Erlacher/No ICA, No SSP, No CSD'

periodogram_avg = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=erlacher_epochs, events=erlacher_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='O_Kniebeugen, Lucid vs. REM vs. Wake', saving_directory=saving_directory)

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Erlacher/No ICA, Yes SSP, No CSD'

periodogram_avg_ssp = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=erlacher_epochs_icassp, events=erlacher_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='O_Kniebeugen, Lucid vs. REM vs. Wake', saving_directory=saving_directory)
#====== Erlacher ==========

#%%====== Jarrot ==========
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Jarrot/No ICA, No SSP, No CSD'

periodogram_avg = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=jarrot_epochs, events=jarrot_events, fmin=0.1, fmax=36, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Jarrot, Lucid vs. REM vs. Wake', saving_directory=saving_directory)

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Jarrot/Yes ICA, Yes SSP, No CSD'

periodogram_avg_icassp = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=jarrot_epochs_icassp, events=jarrot_events, fmin=0.1, fmax=36, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Jarrot, Lucid vs. REM vs. Wake', saving_directory=saving_directory)
    
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Jarrot/Yes ICA, Yes SSP, Yes CSD'

periodogram_avg_icasspcsd = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=jarrot_epochs_icasspcsd, events=jarrot_events, fmin=0.1, fmax=36, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Jarrot, Lucid vs. REM vs. Wake', saving_directory=saving_directory)
#====== Jarrot ==========

#%%====== Sergio ==========
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Sergio/No ICA, No SSP, No CSD'

periodogram_avg = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=sergio_epochs, events=sergio_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Sergio, Lucid vs. REM vs. Wake', saving_directory=saving_directory)

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Sergio/Yes ICA, Yes SSP, No CSD'

periodogram_avg_icassp = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=sergio_epochs_icassp, events=sergio_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Sergio, Lucid vs. REM vs. Wake', saving_directory=saving_directory)
    
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Sergio/Yes ICA, Yes SSP, Yes CSD'

periodogram_avg_icasspcsd = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=sergio_epochs_icasspcsd, events=sergio_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Sergio, Lucid vs. REM vs. Wake', saving_directory=saving_directory)
#====== Sergio ==========

#%%====== Lucireta ==========
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Lucireta/No ICA, No SSP, No CSD'

periodogram_avg = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=lucireta_epochs, events=lucireta_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Lucireta, Lucid vs. REM vs. Wake', saving_directory=saving_directory)

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Lucireta/Yes ICA, Yes SSP, No CSD'

periodogram_avg_icassp = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=lucireta_epochs_icassp, events=lucireta_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Lucireta, Lucid vs. REM vs. Wake', saving_directory=saving_directory)
    
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Lucireta/Yes ICA, Yes SSP, Yes CSD'

periodogram_avg_icasspcsd = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=lucireta_epochs_icasspcsd, events=lucireta_events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Lucireta, Lucid vs. REM vs. Wake', saving_directory=saving_directory)
#====== Lucireta ==========
#%% ===== PSD [%] (ICA+SSP+CSD) One by One 6 channels ========
for i in range(len(jarrot_epochlist_icasspcsd)):
    jarrot_epochlist_icasspcsd[i] = jarrot_epochlist_icasspcsd[i].pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
for i in range(len(sergio_epochlist_icasspcsd)):
    sergio_epochlist_icasspcsd[i] = sergio_epochlist_icasspcsd[i].pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
for i in range(len(lucireta_epochlist_icasspcsd)):
    lucireta_epochlist_icasspcsd[i] = lucireta_epochlist_icasspcsd[i].pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])    
    
merged_individual_files_icasspcsd_6channels = fild_epochslist_icasspcsd + sergio_epochlist_icasspcsd + \
                                              lucireta_epochlist_icasspcsd
                                    
periodogram_list_6channels = list()
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms [%]/One by One/6 Channels'
for i in range(len(merged_individual_files_icasspcsd_6channels)):
    temp_epochs = merged_individual_files_icasspcsd_6channels[i]
    periodogram_avg_icassp, freqs = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=temp_epochs, events=temp_epochs.events, fmin=0.1, fmax=48, \
    n_overlap=128, standardization=True, psd_type='multitaper', explanation='Night_' + str(i), saving_directory=None)
        
    periodogram_list_6channels.append(periodogram_avg_icassp)
    
#============ Merge into one Plot =================
fig, ax = plt.subplots(ncols=3, nrows=7, constrained_layout=True)
# fig.subplots_adjust(hspace=1)
for i in range(len(periodogram_list_6channels)):
    ax_x, ax_y = int(i/3), i%3 #axis finder
    ax[ax_x, ax_y].plot(freqs, periodogram_list_6channels[i][0], color='blue', ls='-', label='REM', linewidth=1)
    ax[ax_x, ax_y].plot(freqs, periodogram_list_6channels[i][1], color='red', ls='-', label='Lucid', linewidth=1)
    ax[ax_x, ax_y].plot(freqs, periodogram_list_6channels[i][2], color='green', ls='-', label='Wake', linewidth=1)
    
    ax[ax_x, ax_y].set_title('Night_' + str(i+1), size=15)
    ax[ax_x, ax_y].legend(loc='upper right', prop={'size': 6, 'weight':3})
    
    ax[ax_x, ax_y].set_xlabel('Frequency (Hz)', size=10)
    ax[ax_x, ax_y].set_ylabel('Power [%]', size=10)
    # ax.legend(loc='upper right')

fig.suptitle('PSD [%] of Each Night of 6 Channels (F3, F4, C3, C4, O1, O2)', size=30)
fig.delaxes(ax[6][1])
fig.delaxes(ax[6][2])

chetto_EEG.save_figure(saving_directory, explanation='PSD [%] of Each Night 6 Channels (F3, F4, C3, C4, O1, O2)', dpi=400)
#============ Merge into one Plot ================

#%% ===== PSD [%] (ICA+SSP+CSD) One by One 2 channels ========
for i in range(len(jarrot_epochlist_icasspcsd)):
    jarrot_epochlist_icasspcsd[i] = jarrot_epochlist_icasspcsd[i].pick_channels(['C3', 'C4'])
for i in range(len(sergio_epochlist_icasspcsd)):
    sergio_epochlist_icasspcsd[i] = sergio_epochlist_icasspcsd[i].pick_channels(['C3', 'C4'])
for i in range(len(lucireta_epochlist_icasspcsd)):
    lucireta_epochlist_icasspcsd[i] = lucireta_epochlist_icasspcsd[i].pick_channels(['C3', 'C4'])    
    
merged_individual_files_icasspcsd_6channels = fild_epochslist_icasspcsd + sergio_epochlist_icasspcsd + \
                                              lucireta_epochlist_icasspcsd + erlacher_epochlist_icassp
                                    
periodogram_list_2channels = list()
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms [%]/One by One/2 Channels'
for i in range(len(merged_individual_files_icasspcsd_6channels)):
    temp_epochs = merged_individual_files_icasspcsd_6channels[i]
    periodogram_avg_icassp, freqs = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=temp_epochs, events=temp_epochs.events, fmin=0.1, fmax=48, \
    n_overlap=128, standardization=True, psd_type='multitaper', explanation='Night_' + str(i), saving_directory=None)
        
    periodogram_list_2channels.append(periodogram_avg_icassp)
    
#============ Merge into one Plot =================
fig, ax = plt.subplots(ncols=3, nrows=9, constrained_layout=True)
# fig.subplots_adjust(hspace=1)
for i in range(len(periodogram_list_2channels)):
    ax_x, ax_y = int(i/3), i%3 #axis finder
    ax[ax_x, ax_y].plot(freqs, periodogram_list_2channels[i][0], color='blue', ls='-', label='REM', linewidth=1)
    ax[ax_x, ax_y].plot(freqs, periodogram_list_2channels[i][1], color='red', ls='-', label='Lucid', linewidth=1)
    ax[ax_x, ax_y].plot(freqs, periodogram_list_2channels[i][2], color='green', ls='-', label='Wake', linewidth=1)
    
    ax[ax_x, ax_y].set_title('Night_' + str(i+1), size=15)
    ax[ax_x, ax_y].legend(loc='upper right', prop={'size': 6, 'weight':3})
    
    ax[ax_x, ax_y].set_xlabel('Frequency (Hz)', size=10)
    ax[ax_x, ax_y].set_ylabel('Power [%]', size=10)
    # ax.legend(loc='upper right')

# fig.subplots_adjust(hspace=1)
fig.suptitle('PSD [%] of Each Night of 2 Channels (C3, C4)', size=30)
fig.delaxes(ax[8][1])
fig.delaxes(ax[8][2])

chetto_EEG.save_figure(saving_directory, explanation='PSD [%] of Each Night\n 2 Channels (C3, C4) ', dpi=400)
#============ Merge into one Plot =================
    
#%%==== PSD Lucid / REM (ICA+SSP+CSD) ========
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD Lucid to Rem Ratio/FILD'
_, LRratio_fild, freqs = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=fild_epochs_icasspcsd, events=fild_events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='FILD', smoothing=True,\
                                                     n_overlap=128, standardization=True, psd_type='multitaper')

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD Lucid to Rem Ratio/Erlacher'
_, LRratio_erlacher,_ = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=erlacher_epochs_icassp, events=erlacher_events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='O_Kniebeugen', smoothing=True,\
                                                     n_overlap=128, standardization=True, psd_type='multitaper')

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD Lucid to Rem Ratio/Jarrot'
_, LRratio_jarrot,_ = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=jarrot_epochs_icasspcsd, events=jarrot_events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='Jarrot', smoothing=True, \
                                                     n_overlap=128, standardization=True, psd_type='multitaper')
    
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD Lucid to Rem Ratio/Sergio'
_, LRratio_sergio,_ = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=sergio_epochs_icasspcsd, events=sergio_epochs_icasspcsd.events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='Sergio', smoothing=True, \
                                                     n_overlap=128, standardization=True, psd_type='multitaper')
    
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD Lucid to Rem Ratio/Lucireta'
_, LRratio_lucireta,_ = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=lucireta_epochs_icasspcsd, events=lucireta_events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='Lucireta', smoothing=True, \
                                                     n_overlap=128, standardization=True, psd_type='multitaper')
    
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD Lucid to Rem Ratio/Lucireta'
_, LRratio_lucireta,_ = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=lucireta_epochs_icasspcsd, events=lucireta_events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='Lucireta', smoothing=True, \
                                                     n_overlap=128, standardization=True, psd_type='multitaper')

#%% ===== PSD Lucid / REM (ICA+SSP+CSD) One by One 6 channels ========
for i in range(len(jarrot_epochlist_icasspcsd)):
    jarrot_epochlist_icasspcsd[i] = jarrot_epochlist_icasspcsd[i].pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
for i in range(len(sergio_epochlist_icasspcsd)):
    sergio_epochlist_icasspcsd[i] = sergio_epochlist_icasspcsd[i].pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
for i in range(len(lucireta_epochlist_icasspcsd)):
    lucireta_epochlist_icasspcsd[i] = lucireta_epochlist_icasspcsd[i].pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])    
    
merged_individual_files_icasspcsd_6channels = fild_epochslist_icasspcsd + jarrot_epochlist_icasspcsd + \
                                              lucireta_epochlist_icasspcsd
                                    
LRratio_list = list()
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD Lucid to Rem Ratio/One by One'
for i in range(len(merged_individual_files_icasspcsd_6channels)):
    temp_epochs = merged_individual_files_icasspcsd_6channels[i]
    _, LRratio_onebyone,_ = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=temp_epochs, events=temp_epochs.events, fmin=0.1, fmax=48, \
                                                         saving_directory=saving_directory, explanation='File_' + str(i), smoothing=True, \
                                                         n_overlap=128, standardization=True, psd_type='multitaper')
    LRratio_list.append(LRratio_onebyone)
    
#%% Continue with MErging into one PLot    
plt.figure()
ax = plt.axes()
import random
for i in range(len(LRratio_list)):
    color = "#%06x" % random.randint(0, 0xFFFFFF)
    ax.plot(freqs, LRratio_list[i], color=color, ls='-', linewidth=3, label='File_'+str(i))  

#Grand AVG
LRratio_grand_avg = np.mean(LRratio_list, axis=0)
ax.plot(freqs, LRratio_grand_avg, color='yellow', ls='-', linewidth=6, label='Grand Avg') 

#==== Frequency Limit Drawer ======
amp_linspace = np.linspace(0.9, 1.1, num=len(freqs))
ax.plot(np.ones(len(freqs)), amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*4, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*8, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*12, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*16, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*20, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*28, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*36, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*45, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
#==== Frequency Limit Drawer ======

#===== Texts ======
ax.text(1.2, 1.03, s='δ (1–4 Hz)', color='black', fontsize=20)
ax.text(4.8, 1.03, s='θ (4–8 Hz)', color='black', fontsize=20)
ax.text(8.5, 1.03, s='α (8–12 Hz)', color='black', fontsize=20)
ax.text(12.2, 1.03, s='β1 (12-16 Hz)', color='black', fontsize=20)
ax.text(16.2, 1.03, s='β2 (16–20 Hz)', color='black', fontsize=20)
ax.text(22.4, 1.03, s='γ1 (20–28 Hz)', color='black', fontsize=20)
ax.text(30.5, 1.03, s='γ2 (28–36 Hz)', color='black', fontsize=20)
ax.text(38.7, 1.03, s='γ-40Hz (36-45 Hz)', color='black', fontsize=20)
ax.text(45.5, 1.03, s='γ+ (45+ Hz)', color='black', fontsize=20)
#===== Texts ======

ax.set_title('Lucid / REM Ratio of Each Night \n6 Channels (F3, F4, C3, C4, O1, O2)', size=25)
ax.set_xlabel('Frequency (Hz)', size=20)
ax.set_ylabel('Power Lucid / REM', size=20)
ax.set_ylim(0.7, 1.4)
ax.legend(loc='upper left', prop={'size': 11, 'weight':3})
ax.plot(freqs, np.ones(len(freqs)), ls='--', linewidth=5, color='black')
#%% ===== PSD MNE Plots ======
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}

chetto_EEG.PSD_of_all_stages(epochs=fild_epochs_icasspcsd, event_id=event_id, picks='all',\
                             explanation= 'FILD Dataset, PSD of Lucidity vs. REM vs. Awake',\
                             saving_directory='/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD MNE Plots/FILD')
    
chetto_EEG.PSD_of_all_stages(epochs=erlacher_epochs_icassp, event_id=event_id, picks='all',\
                             explanation= 'O_Kniebeugen Dataset, PSD of Lucidity vs. REM vs. Awake',\
                             saving_directory='/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD MNE Plots/Erlacher')
    
chetto_EEG.PSD_of_all_stages(epochs=jarrot_epochs_icasspcsd, event_id=event_id, picks='all', fmax=34,\
                             explanation= 'Jarrot Dataset, PSD of Lucidity vs. REM vs. Awake',\
                             saving_directory='/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD MNE Plots/Jarrot')
    
chetto_EEG.PSD_of_all_stages(epochs=sergio_epochs_icasspcsd, event_id=event_id, picks='all',\
                             explanation= 'Sergio Dataset, PSD of Lucidity vs. REM vs. Awake',\
                             saving_directory='/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD MNE Plots/Sergio')
    
chetto_EEG.PSD_of_all_stages(epochs=lucireta_epochs_icasspcsd_rereferenced, event_id=event_id, picks='all',\
                             explanation= 'Lucireta Dataset, PSD of Lucidity vs. REM vs. Awake_rereferenced',\
                             saving_directory='/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD MNE Plots/Lucireta')
#%% ========== Merge ICA + SSP + CSD Overall Lucid / REM Ratio ==========
plt.figure()
ax = plt.axes()
color_chunk = ['#ef3e48', '#ffdf14', '#1fb887', '#afab4e', '#b614b8', '#76283c']

intervalLength = 31
LRratio_fild_env = chetto_EEG.envelopeCreator(LRratio_fild, intervalLength=intervalLength)
LRratio_erlacher_env = chetto_EEG.envelopeCreator(LRratio_erlacher,intervalLength=intervalLength)
LRratio_jarrot_env = chetto_EEG.envelopeCreator(LRratio_jarrot,intervalLength=intervalLength)
# LRratio_sergio_env = chetto_EEG.envelopeCreator(LRratio_sergio,intervalLength=intervalLength)
LRratio_lucireta_env = chetto_EEG.envelopeCreator(LRratio_lucireta,intervalLength=intervalLength)
meanofAll = np.zeros((4,191))
meanofAll[0] = LRratio_fild
meanofAll[1] = LRratio_erlacher
meanofAll[2] = LRratio_jarrot
# meanofAll[3] = LRratio_sergio
meanofAll[3] = LRratio_lucireta
meanofAll = np.mean(meanofAll,0)
LRratio_meanofAll_env = chetto_EEG.envelopeCreator(meanofAll,intervalLength=intervalLength)

ax.plot(freqs, LRratio_fild_env, color=color_chunk[0], ls='-', linewidth=3, label='FILD')
ax.plot(freqs, LRratio_erlacher_env, color=color_chunk[1], ls='-', linewidth=3, label='O_Kniebeugen')
ax.plot(freqs, LRratio_jarrot_env, color=color_chunk[2], ls='-', linewidth=3, label='Jarrot')
# ax.plot(freqs, LRratio_sergio_env, color=color_chunk[3], ls='-', linewidth=3, label='Sergio')
ax.plot(freqs, LRratio_lucireta_env, color=color_chunk[4], ls='-', linewidth=3, label='Lucireta')
ax.plot(freqs, LRratio_meanofAll_env, color=color_chunk[5], ls='-', linewidth=7, label='Average')

grand_min, grand_max = min(min(LRratio_fild), min(LRratio_erlacher), min(LRratio_jarrot), min(LRratio_sergio), min(LRratio_lucireta)),\
                       max(max(LRratio_fild), max(LRratio_erlacher), max(LRratio_jarrot), max(LRratio_sergio), max(LRratio_lucireta))
#==== Frequency Limit Drawer ======
amp_linspace = np.linspace(grand_min, grand_max, num=len(freqs))
ax.plot(np.ones(len(freqs)), amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*4, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*8, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*12, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*16, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*20, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*28, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*36, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
ax.plot(np.ones(len(freqs))*45, amp_linspace, ls='--',  linewidth=3, color='#5f6567')
#==== Frequency Limit Drawer ======

#===== Texts ======
ax.text(1.2, 1.03, s='δ (1–4 Hz)', color='black', fontsize=20)
ax.text(4.8, 1.03, s='θ (4–8 Hz)', color='black', fontsize=20)
ax.text(8.5, 1.03, s='α (8–12 Hz)', color='black', fontsize=20)
ax.text(12.2, 1.03, s='β1 (12-16 Hz)', color='black', fontsize=20)
ax.text(16.2, 1.03, s='β2 (16–20 Hz)', color='black', fontsize=20)
ax.text(22.4, 1.03, s='γ1 (20–28 Hz)', color='black', fontsize=20)
ax.text(30.5, 1.03, s='γ2 (28–36 Hz)', color='black', fontsize=20)
ax.text(38.7, 1.03, s='γ-40Hz (36-45 Hz)', color='black', fontsize=20)
ax.text(45.5, 1.03, s='γ+ (45+ Hz)', color='black', fontsize=20)
#===== Texts ======

ax.set_title('Lucid / REM Ratio of All Datasets', size=25)
ax.set_xlabel('Frequency (Hz)', size=20)
ax.set_ylabel('Power Lucid / REM', size=20)
ax.legend(loc='upper left', prop={'size': 20, 'weight':3})
ax.plot(freqs, np.ones(len(freqs)), ls='--', linewidth=5, color='black')
plt.show()

#%% ==================== 6 Channels Merged Periodogram, Lucid / REM =============
sergio_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/sergio_epochs_events','rb'))
fild_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/fild_epochs_events','rb'))
jarrot_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/jarrot_epochs_events','rb'))
lucireta_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/lucireta_epochs_events_final','rb'))

jarrot_picked = jarrot_epochs_icacsd.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
sergio_picked = sergio_epochs_icacsd.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
lucireta_picked = lucireta_epochs_icacsd.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
merged_icacsd_6channels = chetto_EEG.epoch_concatenation([fild_epochs_icacsd, sergio_picked, lucireta_picked])

#Periodogram
# saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Merged/6 Channels'
# periodogram_avg_icasspcsd = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=merged_icacsd, events=merged_icacsd.events, fmin=0.1, fmax=48, \
# n_overlap=128, standardization=True, psd_type='multitaper', explanation='6 Channels (F3, F4, C3, C4, O1, O2) Merged, Lucid vs. REM vs. Wake',\
# saving_directory=saving_directory, channelbychannel=True)

#Lucid / REM
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD Lucid to Rem Ratio/Merged'
_, LRratio_fild, freqs = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=merged_icacsd_6channels, events=merged_icacsd_6channels.events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, \
                                                     explanation='6 Channels (F3, F4, C3, C4, O1, O2) Merged', smoothing=True,\
                                                     n_overlap=128, standardization=True, psd_type='multitaper')
#%% ================ 2 Channels Merged Periodogram, Lucid / REM ================
sergio_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/sergio_epochs_events','rb'))
fild_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/fild_epochs_events','rb'))
jarrot_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/jarrot_epochs_events','rb'))
erlacher_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/o_kniebeugen_epochs_events','rb'))
lucireta_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/lucireta_epochs_events_final','rb'))

fild_picked = fild_epochs_icacsd.pick_channels(['C3', 'C4'])
sergio_picked = sergio_epochs_icacsd.pick_channels(['C3', 'C4'])
lucireta_picked = lucireta_epochs_icacsd.pick_channels(['C3', 'C4'])
merged_icacsd_2channels = chetto_EEG.epoch_concatenation([fild_picked, sergio_picked, erlacher_epochs_icacsd, lucireta_picked])

#Periodogram
# saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Periodograms/Merged/2 Channels'
# periodogram_avg_icasspcsd = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=merged_icacsd, events=merged_icacsd.events, fmin=0.1, fmax=48, \
# n_overlap=128, standardization=True, psd_type='multitaper', explanation='2 Channels (C3, C4) Merged, Lucid vs. REM vs. Wake',\
# saving_directory=saving_directory, channelbychannel=True)

#Lucid / REM
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD Lucid to Rem Ratio/Merged'
_, LRratio_fild, freqs = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=merged_icacsd_2channels, events=merged_icacsd_2channels.events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, \
                                                     explanation='2 Channels (C3, C4) Merged', smoothing=True,\
                                                     n_overlap=128, standardization=True, psd_type='multitaper')
#%% ============ Other Analysis ==========
from mne.time_frequency import tfr_multitaper

merged_icacsd_6channels['Lucid'].plot_psd_topomap(ch_type='csd', dB=True, proj=True)
merged_icacsd_6channels['Lucid'].plot_image(picks='csd', combine='mean')
merged_icacsd_6channels['Lucid'].plot_psd(picks='csd')
event_id = {'REM' :0, 'Lucid': 1, 'Wake': 2}

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD MNE Plots/6 Channels Merged'
chetto_EEG.PSD_of_all_stages(epochs=merged_icacsd_6channels, event_id=event_id, explanation='6 Channels (F3, F4, C3, C4, O1, O2) Merged \nPSD of All Stages', picks='csd', saving_directory=saving_directory)

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/PSD MNE Plots/2 Channels Merged'
chetto_EEG.PSD_of_all_stages(epochs=merged_icacsd_2channels, event_id=event_id, explanation='2 Channels (C3, C4) Merged \nPSD of All Stages', picks='csd', saving_directory=saving_directory)

#Multi-taper Parameters
freqs = np.arange(0.1, 48, 0.1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
n_cycles = freqs * 2  # use constant t/f resolution

#========= 6 Channels ==========
fig, ax = plt.subplots(3,1, figsize=(12, 4)) #gridspec_kw={"width_ratios": [5, 5]})

merged_icacsd_6channels.load_data()
power = tfr_multitaper(merged_icacsd_6channels['Lucid'], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, time_bandwidth = 8.0, decim=2,\
                       average=True)
power.apply_baseline([-0.2, 0], mode='mean')
avg_power = np.mean(power._data, 0)
power._data = np.expand_dims(avg_power, axis=0) #expand dimension from axis=0 [1,x,y]
power.plot([0], vmin=vmin, vmax=vmax, axes=ax[0], colorbar=False, show=False)

power = tfr_multitaper(merged_icacsd_6channels['REM'], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, time_bandwidth = 8.0, decim=2,\
                       average=True)
power.apply_baseline([-0.2, 0], mode='mean')
avg_power = np.mean(power._data, 0)
power._data = np.expand_dims(avg_power, axis=0) #expand dimension from axis=0 [1,x,y]
power.plot([0], vmin=vmin, vmax=vmax, axes=ax[1], colorbar=False, show=False)

power = tfr_multitaper(merged_icacsd_6channels['Wake'], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, time_bandwidth = 8.0, decim=2,\
                       average=True)
power.apply_baseline([-0.2, 0], mode='mean')
avg_power = np.mean(power._data, 0)
power._data = np.expand_dims(avg_power, axis=0) #expand dimension from axis=0 [1,x,y]
power.plot([0], vmin=vmin, vmax=vmax, axes=ax[2], colorbar=False, show=False)
#========= 6 Channels ==========

#========= 2 Channels ==========
merged_icacsd_2channels.load_data()
power = tfr_multitaper(merged_icacsd_2channels['Lucid'], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, time_bandwidth = 8.0, decim=2,\
                       average=True)
power.apply_baseline([-0.2, 0], mode='mean')
avg_power = np.mean(power._data, 0)
power._data = np.expand_dims(avg_power, axis=0) #expand dimension from axis=0 [1,x,y]
power.plot([0], vmin=vmin, vmax=vmax, axes=ax[0], colorbar=False, show=False)

power = tfr_multitaper(merged_icacsd_2channels['REM'], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, time_bandwidth = 8.0, decim=2,\
                       average=True)
power.apply_baseline([-0.2, 0], mode='mean')
avg_power = np.mean(power._data, 0)
power._data = np.expand_dims(avg_power, axis=0) #expand dimension from axis=0 [1,x,y]
power.plot([0], vmin=vmin, vmax=vmax, axes=ax[0], colorbar=False, show=False)

power = tfr_multitaper(merged_icacsd_2channels['Wake'], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, time_bandwidth = 8.0, decim=2,\
                       average=True)
power.apply_baseline([-0.2, 0], mode='mean')
avg_power = np.mean(power._data, 0)
power._data = np.expand_dims(avg_power, axis=0) #expand dimension from axis=0 [1,x,y]
power.plot([0], vmin=vmin, vmax=vmax, axes=ax[0], colorbar=False, show=False)
#========= 2 Channels ==========

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
                                        [[27150, 27256.5],[28197.5, 28304],[1108, 1215]]])
                                        # [[23910, 24000],[24696.5, 24839.5],[909, 1052]]])
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
                                  [[27513, 27801],[40077, 40182]],[7506, 7722]], dtype=object)    
    
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

#%% ============== Spectrogram Analysis of Neighbourhood Data (FILD) =====================

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Spectrograms/Lucid & REM & Wake/FILD'

FILD_neighbour_interval, FILD_epochs, FILD_periods = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval/FILD_lucidneighbour_interval','rb'))

powers = chetto_EEG.multitaper_spectrogram_lucidity_neighbours(epochs=FILD_epochs, lucidity_periods=lucidity_period_fild, explanation='FILD Spectrogram of REM & Lucid & Wake',\
                                                               saving_directory=saving_directory)
#%% ============== Spectrogram Analysis of Neighbourhood Data (Erlacher) =====================

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Spectrograms/Lucid & REM & Wake/Erlacher'

erlacher_neighbour_interval, erlacher_epochs, erlacher_periods = \
pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval/Erlacher_lucidneighbour_interval','rb'))

powers = chetto_EEG.multitaper_spectrogram_lucidity_neighbours(epochs=erlacher_epochs, lucidity_periods=lucidity_period_okniebeugen, explanation='O_Knienebeugen Spectrogram of REM & Lucid & Wake',\
                                                               saving_directory=saving_directory)
#%% ============== Spectrogram Analysis of Neighbourhood Data (Sergio) =====================

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Spectrograms/Lucid & REM & Wake/Sergio'

sergio_neighbour_interval, sergio_epochs, sergio_periods = \
pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval/Sergio_lucidneighbour_interval','rb'))

powers = chetto_EEG.multitaper_spectrogram_lucidity_neighbours(epochs=sergio_epochs, lucidity_periods=lucidity_period_sergio, explanation='Sergio Spectrogram of REM & Lucid & Wake',\
                                                               saving_directory=saving_directory)
#%% ============== Spectrogram Analysis of Neighbourhood Data (Jarrot) =====================

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Spectrograms/Lucid & REM & Wake/Jarrot'

jarrot_neighbour_interval, jarrot_epochs, jarrot_periods = \
pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval/Jarrot_lucidneighbour_interval','rb'))

powers = chetto_EEG.multitaper_spectrogram_lucidity_neighbours(epochs=jarrot_epochs, lucidity_periods=lucidity_period_jarrod, explanation='Jarrot Spectrogram of REM & Lucid & Wake',\
                                                               saving_directory=saving_directory, fmax=35)
#%% ============== Spectrogram Analysis of Neighbourhood Data (Lucireta) =====================
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Spectrograms/Lucid & REM & Wake/Lucireta/new'

lucireta_epochs, lucireta_periods = \
pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Lucid Neighbour Interval/Lucireta_lucidneighbour_interval','rb'))


powers = chetto_EEG.multitaper_spectrogram_lucidity_neighbours(epochs=lucireta_epochs[8:], lucidity_periods=lucidity_period_munich[8:], explanation='Lucireta Spectrogram of REM & Lucid & Wake',\
                                                               saving_directory=saving_directory)
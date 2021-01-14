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

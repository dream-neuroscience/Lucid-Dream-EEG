# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:36:22 2020

@author: Cagatay Demirel
"""

import extremeEEGSignalAnalyzer as chettoEEG
#%% ======= Overall EDF Decomposition ===========
writing_directory = 'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/EDF Decomposition'
multiple_folders = ('C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/2_Christian_Tausch/O_Kniebeugen', \
                    'C:/Users/caghangir/Desktop/PhD/Research/Lucid Dream EEG/Total Data/FILD_Raw_data', \
                    'C:/Users/caghangir/Desktop/PhD\Research/Lucid Dream EEG/Total Data/LD_EEG_Jarrot', \
                    'C:/Users/caghangir/Desktop/PhD\Research/Lucid Dream EEG/Total Data/Sergio')
allInfo, overallUniqueDataChannels, common_channels_all_folders, overall_common_channels_all_folders = \
chettoEEG.EDFDecomposer(multipleFolders=multiple_folders, writing_directory=writing_directory)
#%% ======= All Info Decomposition ===========
dataChannels_Osnanburch = allInfo[0][1]
dataChannels_O_Kniebeugen = allInfo[1][1]
dataChannels_jarrot = allInfo[2][1]
dataChannels_sergio = allInfo[3][1]
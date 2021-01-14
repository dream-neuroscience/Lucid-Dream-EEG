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
#%% ================ Unsupervised Clustering REM & Lucid & Wake =============
from scipy.signal import welch, periodogram

#======= 6 Channels ==========
os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set')

sergio_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/sergio_epochs_events','rb'))
fild_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/fild_epochs_events','rb'))
jarrot_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/jarrot_epochs_events','rb'))
lucireta_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/lucireta_epochs_events_final','rb'))

jarrot_picked = jarrot_epochs_icacsd.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
# sergio_picked = sergio_epochs_icacsd.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
lucireta_picked = lucireta_epochs_icacsd.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
merged_icacsd = chetto_EEG.epoch_concatenation([fild_epochs_icacsd, jarrot_picked, lucireta_picked])

trainX_raw = merged_icacsd._data
trainY = merged_icacsd.events[:,2]

trainX = chetto_EEG.featureExtraction(X_train=trainX_raw, Fs=100, window='hann', prenorm=True, featurenorm=False)
pickle.dump([trainX, trainY],open('featureSet_6channels_prenormed_nofeaturenorm','wb'))
#======= 6 Channels ==========

#%%======= 2 Channels =========
os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set')

sergio_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/sergio_epochs_events','rb'))
fild_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/fild_epochs_events','rb'))
jarrot_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/jarrot_epochs_events','rb'))
erlacher_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/o_kniebeugen_epochs_events','rb'))
lucireta_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/lucireta_epochs_events_final','rb'))

fild_picked = fild_epochs_icacsd.pick_channels(['C3', 'C4'])
sergio_picked = sergio_epochs_icacsd.pick_channels(['C3', 'C4'])
lucireta_picked = lucireta_epochs_icacsd.pick_channels(['C3', 'C4'])
merged_icacsd = chetto_EEG.epoch_concatenation([fild_picked, sergio_picked, erlacher_epochs_icacsd, lucireta_picked])

trainX_raw = merged_icacsd._data
trainY = merged_icacsd.events[:,2]

trainX = chetto_EEG.featureExtraction(X_train=trainX_raw, Fs=100, window='hann', prenorm=True, featurenorm=True)
pickle.dump([trainX, trainY],open('featureSet_2channels_prenormed_featurenorm','wb'))
#======= 2 Channels =========
#%% ============== Lucireta Feature Extraction ==============
os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set')

lucireta_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/lucireta_epochs_events_final','rb'))

trainX_raw = lucireta_epochs_icacsd._data
trainY = lucireta_epochs_icacsd.events[:,2]

trainX = chetto_EEG.featureExtraction(X_train=trainX_raw, Fs=100, window='hann', prenorm=False, featurenorm=True)
pickle.dump([trainX, trainY],open('featureSet_lucireta_noprenormed_featurenorm','wb'))
#%% ============== FILD Feature Extraction ==============
os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set')

fild_epochs_icacsd, _ = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes/fild_epochs_events_nobaseline','rb'))

trainX_raw = fild_epochs_icacsd._data
trainY = fild_epochs_icacsd.events[:,2]

baselineinterval = trainX_raw[0,0]
trainX = chetto_EEG.featureExtraction(X_train=trainX_raw, Fs=100, window='hann', prenorm=True, featurenorm=True, \
                                      baselineinterval=baselineinterval)
pickle.dump([trainX, trainY],open('featureSet_fild_noprenormed_featurenorm_globbaselin','wb'))
#%% ============= Unsupervised & Semi-supervised Clustering 6 channels Quick Run ================
saving_directory_FV = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/6 Channels/PreNormed/FeatureNormed/Feature Visualization'
saving_directory_CS = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/6 Channels/PreNormed/FeatureNormed/Cluster Scatters'
saving_results = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/6 Channels/PreNormed/FeatureNormed/Results'
explanation = 'REM & Lucid & Awake'
class_names = ('REM', 'Lucid', 'Awake')

trainX, trainY = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set/featureSet_6channels_prenormed_featurenorm','rb'))
# trainX, trainY = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set/featureSet_2channels','rb'))

results_6channels, trainX_cleaned_6channels = chetto_ML.unsupervised_semisupervised_quick_run(feature_x=trainX, feature_y=trainY, class_amount=3, custom_dimension=None, \
                                              saving_directory_FV=saving_directory_FV, saving_directory_CS=saving_directory_CS, \
                                              visualization=True, saving_results=saving_results, \
                                              skip_first_step=False, supervised_dim_reduction=False, explanation=explanation, \
                                              class_names=class_names, max_depth=100)
#%% ============= Unsupervised & Semi-supervised Clustering 2 channels Quick Run ================
saving_directory_FV = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/2 Channels/PreNormed/FeatureNormed/Feature Visualization'
saving_directory_CS = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/2 Channels/PreNormed/FeatureNormed/Cluster Scatters'
saving_results = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/2 Channels/PreNormed/FeatureNormed/Results'
explanation = 'REM & Lucid & Awake'
class_names = ('REM', 'Lucid', 'Awake')

trainX, trainY = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set/featureSet_2channels_prenormed_featurenorm','rb'))
# trainX, trainY = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set/featureSet_2channels','rb'))

results_2channels = chetto_ML.unsupervised_semisupervised_quick_run(feature_x=trainX, feature_y=trainY, class_amount=3, custom_dimension=None, \
                                              saving_directory_FV=saving_directory_FV, saving_directory_CS=saving_directory_CS, \
                                              visualization=True, saving_results=saving_results, \
                                              skip_first_step=False, supervised_dim_reduction=False, explanation=explanation, \
                                              class_names=class_names)
#%% ============= Unsupervised & Semi-supervised Clustering Lucireta Quick Run ================
saving_directory_FV = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/Lucireta/NoPreNorm/FeatureNormed/Feature Visualization'
saving_directory_CS = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/Lucireta/NoPreNorm/FeatureNormed/Cluster Scatters'
saving_results = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/Lucireta/NoPreNorm/FeatureNormed/Results'
explanation = 'REM & Lucid & Awake'
class_names = ('REM', 'Lucid', 'Awake')

trainX, trainY = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set/featureSet_lucireta_noprenormed_featurenorm','rb'))
# trainX, trainY = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set/featureSet_2channels','rb'))

results_lucireta = chetto_ML.unsupervised_semisupervised_quick_run(feature_x=trainX, feature_y=trainY, class_amount=3, custom_dimension=None, \
                                              saving_directory_FV=saving_directory_FV, saving_directory_CS=saving_directory_CS, \
                                              visualization=True, saving_results=saving_results, \
                                              skip_first_step=False, supervised_dim_reduction=False, explanation=explanation, \
                                              class_names=class_names)
#%% ============= Unsupervised & Semi-supervised Clustering FILD Quick Run ================
saving_directory_FV = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/FILD_globalbaseline/PreNormed/FeatureNormed/Feature Visualization'
saving_directory_CS = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/FILD_globalbaseline/PreNormed/FeatureNormed/Cluster Scatters'
saving_results = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/FILD_globalbaseline/PreNormed/FeatureNormed/Results'
explanation = 'REM & Lucid & Awake'
class_names = ('REM', 'Lucid', 'Awake')

trainX, trainY = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set/featureSet_fild_noprenormed_featurenorm_globbaselin','rb'))
# trainX, trainY = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Feature Set/featureSet_2channels','rb'))

results_fild, trainX_cleaned_fild = chetto_ML.unsupervised_semisupervised_quick_run(feature_x=trainX, feature_y=trainY, class_amount=3, custom_dimension=None, \
                                              saving_directory_FV=saving_directory_FV, saving_directory_CS=saving_directory_CS, \
                                              visualization=True, saving_results=saving_results, \
                                              skip_first_step=False, supervised_dim_reduction=False, explanation=explanation, \
                                              class_names=class_names, max_depth=100)
#%% ================ Result Analysis ==========
trainX_6channel_nofeatnorm_2d, trainX_6channel_nofeatnorm_3d, trainX_6channel_nofeatnorm_allD = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/6 Channels/PreNormed/FeatureNormed/Results/REM & Lucid & Awake_total_results_kmeans_2&3D','rb'))
trainX_6channel_featnorm_2d, trainX_6channel_featnorm_3d, trainX_6channel_featnorm_allD = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/6 Channels/PreNormed/NoFeatureNormed/Results/REM & Lucid & Awake_total_results_kmeans_2&3D','rb'))
trainX_2channel_nofeatnorm_2d, trainX_2channel_nofeatnorm_3d, trainX_2channel_nofeatnorm_allD = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/2 Channels/PreNormed/FeatureNormed/Results/REM & Lucid & Awake_total_results_kmeans_2&3D','rb'))
trainX_2channel_featnorm_2d, trainX_2channel_featnorm_3d, trainX_2channel_featnorm_allD = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/2 Channels/PreNormed/NoFeatureNormed/Results/REM & Lucid & Awake_total_results_kmeans_2&3D','rb'))

trainX_lucireta_prenormfeatnorm_2d, trainX_lucireta_prenormfeatnorm_3d, trainX_lucireta_prenormfeatnorm_allD = \
pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/Lucireta/PreNormed/FeatureNormed/Results/REM & Lucid & Awake_total_results_kmeans_2&3D','rb'))

trainX_lucireta_prenormnofeatnorm_2d, trainX_lucireta_prenormnofeatnorm_3d, trainX_lucireta_prenormnofeatnorm_allD = \
pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/Lucireta/PreNormed/NoFeatureNormed/Results/REM & Lucid & Awake_total_results_kmeans_2&3D','rb'))

trainX_lucireta_noprenormfeatnorm_2d, trainX_lucireta_noprenormfeatnorm_3d, trainX_lucireta_noprenormfeatnorm_allD = \
pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/Lucireta/NoPreNorm/FeatureNormed/Results/REM & Lucid & Awake_total_results_kmeans_2&3D','rb'))

accuracy_yespreyesfeat = np.mean(trainX_lucireta_prenormfeatnorm_2d['accuracy'])
accuracy_yesprenofeat = np.mean(trainX_lucireta_prenormnofeatnorm_2d['accuracy'])
accuracy_nopreyesfeat = np.mean(trainX_lucireta_noprenormfeatnorm_2d['accuracy'])

#FILD
trainX_fild_prenormfeatnorm_2d, trainX_fild_prenormfeatnorm_3d, trainX_fild_prenormfeatnorm_allD = \
pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/FILD/PreNorm/FeatureNormed/Results/REM & Lucid & Awake_total_results_kmeans_2&3D','rb'))

trainX_fild_noprenormfeatnormglobbas_2d, trainX_fild_noprenormfeatnormglobbas_3d, trainX_fild_noprenormfeatnormglobbas_allD = \
pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/Unsupervised/FILD_globalbaseline/NoPreNorm/FeatureNormed/Results/REM & Lucid & Awake_total_results_kmeans_2&3D','rb'))
#%% ============ Area 51 ============

ranks, chosen_features, feature_set_selected = chetto_ML.feature_selection_Boruta(X=trainX, y=trainY, max_depth=100)   
ordered_ranks_indexes = np.argsort(-1 * ranks) #choose first 4 features as important
ordered_ranks = ranks[ordered_ranks_indexes] 
#=== Feature Selection =====

#==== Outlier Cleaning x1=====
upper_limit = min(np.argwhere(ordered_ranks == min(ordered_ranks))[0])
outlier_sample_indexes, cleaned_feature_x, cleaned_feature_y = chetto_ML.iqr_based_combinatory_outlier_cleaner(\
                                         feature_set = feature_x, upper_limit=upper_limit, \
                                         feature_y=feature_y)
cleaned_feature_x = cleaned_feature_x[:,chosen_features]
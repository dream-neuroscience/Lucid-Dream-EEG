import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import extremeEEGSignalAnalyzer as chetto_EEG
import extremeMachineLearning as chetto_ML
import pickle
from scipy.stats import zscore

# from mne.preprocessing import ICA
chetto_EEG = chetto_EEG.extremeEEGSignalAnalyzer()
chetto_ML = chetto_ML.extremeMachineLearning()
# %matplotlib qt
# os.chdir('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP Yes, CSD Yes')
#%% ================== Dataset Creation for Statistics ============
import pandas as pd
def LD_dataset_creation_forstatistic(data, freqs, saving_directory, explanation):
    
    freq_bands = np.zeros((6,2)).astype(int)
    freq_bands[0,0], freq_bands[0,1] = min(np.argwhere(freqs >= 0.1))[0], max(np.argwhere(freqs < 4))[0]
    freq_bands[1,0], freq_bands[1,1] = min(np.argwhere(freqs >= 4))[0], max(np.argwhere(freqs < 8))[0]
    freq_bands[2,0], freq_bands[2,1] = min(np.argwhere(freqs >= 8))[0], max(np.argwhere(freqs < 12))[0]
    freq_bands[3,0], freq_bands[3,1] = min(np.argwhere(freqs >= 12))[0], max(np.argwhere(freqs < 30))[0]
    freq_bands[4,0], freq_bands[4,1] = min(np.argwhere(freqs >= 30))[0], max(np.argwhere(freqs < 40))[0]
    freq_bands[5,0], freq_bands[5,1] = min(np.argwhere(freqs >= 40))[0], max(np.argwhere(freqs < 48))[0]
    
    differences = freq_bands[:,1] - freq_bands[:,0] + 1
    
    data_flatten = np.concatenate((data[0], data[1], data[2]))
    data_flatten_standardized = (data_flatten - np.mean(data_flatten)) / np.std(data_flatten)
    
    length = np.size(data,1)
    df = pd.DataFrame({'State': np.repeat(['REM', 'Lucid', 'Wake'], length),
                       'Frequency Band': np.concatenate((np.repeat(['Delta'],differences[0]), 
                                                         np.repeat(['Theta'], differences[1]),
                                                         np.repeat(['Alpha'], differences[2]),
                                                         np.repeat(['Beta'], differences[3]),
                                                         np.repeat(['Gamma -40 Hz'], differences[4]),
                                                         np.repeat(['Gamma +40 Hz'], differences[5]),
                                                         np.repeat(['Delta'],differences[0]), 
                                                         np.repeat(['Theta'], differences[1]),
                                                         np.repeat(['Alpha'], differences[2]),
                                                         np.repeat(['Beta'], differences[3]),
                                                         np.repeat(['Gamma -40 Hz'], differences[4]),
                                                         np.repeat(['Gamma +40 Hz'], differences[5]),
                                                         np.repeat(['Delta'],differences[0]), 
                                                         np.repeat(['Theta'], differences[1]),
                                                         np.repeat(['Alpha'], differences[2]),
                                                         np.repeat(['Beta'], differences[3]),
                                                         np.repeat(['Gamma -40 Hz'], differences[4]),
                                                         np.repeat(['Gamma +40 Hz'], differences[5]))),
                                                         # np.repeat(['y-40 Hz'], differences[5]))),
                       'Periodogram [%]' : data_flatten,
                       'Periodogram [%] Standardized' : data_flatten_standardized
                       })
    
    df.to_csv(saving_directory + '/' + explanation, index=False)
    
def LD_dataset_creation_forRMA(data, freqs, saving_directory, explanation):
    
    freq_bands = np.zeros((6,2)).astype(int)
    freq_bands[0,0], freq_bands[0,1] = min(np.argwhere(freqs >= 0.1))[0], max(np.argwhere(freqs < 4))[0]
    freq_bands[1,0], freq_bands[1,1] = min(np.argwhere(freqs >= 4))[0], max(np.argwhere(freqs < 8))[0]
    freq_bands[2,0], freq_bands[2,1] = min(np.argwhere(freqs >= 8))[0], max(np.argwhere(freqs < 12))[0]
    freq_bands[3,0], freq_bands[3,1] = min(np.argwhere(freqs >= 12))[0], max(np.argwhere(freqs < 30))[0]
    freq_bands[4,0], freq_bands[4,1] = min(np.argwhere(freqs >= 30))[0], max(np.argwhere(freqs < 40))[0]
    freq_bands[5,0], freq_bands[5,1] = min(np.argwhere(freqs >= 40))[0], max(np.argwhere(freqs < 48))[0]
    
    # differences = freq_bands[:,1] - freq_bands[:,0] + 1
    
    # data_flatten = np.concatenate((data[0], data[1], data[2]))
    data = (data - np.mean(data)) / np.std(data)
    
    # length = np.size(data,1)
    dictionary = {'Delta-REM' : data[0][freq_bands[0,0]:freq_bands[0,1]+1],
                  'Theta-REM' : data[0][freq_bands[1,0]:freq_bands[1,1]+1],
                  'Alpha-REM' : data[0][freq_bands[2,0]:freq_bands[2,1]+1],
                  'Beta-REM' : data[0][freq_bands[3,0]:freq_bands[3,1]+1],
                  'lgamma-REM' : data[0][freq_bands[4,0]:freq_bands[4,1]+1],
                  'hgamma-REM' : data[0][freq_bands[5,0]:freq_bands[5,1]+1],
                
                  'Delta-Lucid' : data[1][freq_bands[0,0]:freq_bands[0,1]+1],
                  'Theta-Lucid' : data[1][freq_bands[1,0]:freq_bands[1,1]+1],
                  'Alpha-Lucid' : data[1][freq_bands[2,0]:freq_bands[2,1]+1],
                  'Beta-Lucid' : data[1][freq_bands[3,0]:freq_bands[3,1]+1],
                  'lgamma-Lucid' : data[1][freq_bands[4,0]:freq_bands[4,1]+1],
                  'hgamma-Lucid' : data[1][freq_bands[5,0]:freq_bands[5,1]+1],
                
                  'Delta-Wake' : data[2][freq_bands[0,0]:freq_bands[0,1]+1],
                  'Theta-Wake' : data[2][freq_bands[1,0]:freq_bands[1,1]+1],
                  'Alpha-Wake' : data[2][freq_bands[2,0]:freq_bands[2,1]+1],
                  'Beta-Wake' : data[2][freq_bands[3,0]:freq_bands[3,1]+1],
                  'lgamma-Wake' : data[2][freq_bands[4,0]:freq_bands[4,1]+1],
                  'hgamma-Wake' : data[2][freq_bands[5,0]:freq_bands[5,1]+1]
                       }
    df = pd.DataFrame.from_dict(dictionary, orient='index')
    df=df.transpose()
    
    df=df.fillna(method ='bfill')
    
    df.to_csv(saving_directory + '/' + explanation, index=False)
    
def LD_dataset_creation_ROIs_3regions_forANOVA(data, freqs, saving_directory, explanation):
    
    freq_bands = np.zeros((7,2)).astype(int)
    freq_bands[0,0], freq_bands[0,1] = min(np.argwhere(freqs >= 0.1))[0], max(np.argwhere(freqs < 4))[0]
    freq_bands[1,0], freq_bands[1,1] = min(np.argwhere(freqs >= 4))[0], max(np.argwhere(freqs < 8))[0]
    freq_bands[2,0], freq_bands[2,1] = min(np.argwhere(freqs >= 8))[0], max(np.argwhere(freqs < 12))[0]
    freq_bands[3,0], freq_bands[3,1] = min(np.argwhere(freqs >= 12))[0], max(np.argwhere(freqs < 30))[0]
    freq_bands[4,0], freq_bands[4,1] = min(np.argwhere(freqs >= 30))[0], max(np.argwhere(freqs < 40))[0]
    freq_bands[5,0], freq_bands[5,1] = min(np.argwhere(freqs >= 40))[0], max(np.argwhere(freqs < 45))[0]
    freq_bands[6,0], freq_bands[6,1] = min(np.argwhere(freqs >= 45))[0], max(np.argwhere(freqs < 48))[0]
    
    data = (data - np.mean(data)) / np.std(data)
    
    states = list()
    for i in range(7): #frequency
        state = np.empty(shape=[0,1])    
        for j in range(len(data)): #ROI
            state = np.append(state, np.ones((len(data[0][freq_bands[i,0]:freq_bands[i,1]+1]))) * (j+1))
        states.append(state)
    
    dictionary = dict()
    for i in range(7): #frequency
        dictionary['state' + str(i+1)] = states[i]
        dictionary['frequency' + str(i+1)] = data[:, freq_bands[i,0]:freq_bands[i,1]+1].flatten()
            
    df = pd.DataFrame.from_dict(dictionary, orient='index')
    df=df.transpose()
    
    df=df.fillna(method ='bfill')
    
    df.to_csv(saving_directory + '/' + explanation, index=False)
#%% LR Ratio Statistic    
def LD_dataset_creation_forRMA_LRratio(data, freqs, saving_directory, explanation):
    
    freq_bands = np.zeros((7,2)).astype(int)
    freq_bands[0,0], freq_bands[0,1] = min(np.argwhere(freqs >= 1))[0], max(np.argwhere(freqs < 4))[0]
    freq_bands[1,0], freq_bands[1,1] = min(np.argwhere(freqs >= 4))[0], max(np.argwhere(freqs < 8))[0]
    freq_bands[2,0], freq_bands[2,1] = min(np.argwhere(freqs >= 8))[0], max(np.argwhere(freqs < 12))[0]
    freq_bands[3,0], freq_bands[3,1] = min(np.argwhere(freqs >= 12))[0], max(np.argwhere(freqs < 30))[0]
    freq_bands[4,0], freq_bands[4,1] = min(np.argwhere(freqs >= 30))[0], max(np.argwhere(freqs < 40))[0]
    freq_bands[5,0], freq_bands[5,1] = min(np.argwhere(freqs >= 40))[0], max(np.argwhere(freqs < 45))[0]
    freq_bands[6,0], freq_bands[6,1] = min(np.argwhere(freqs >= 45))[0], max(np.argwhere(freqs < 48))[0]
    
    # differences = freq_bands[:,1] - freq_bands[:,0] + 1
    
    # data_flatten = np.concatenate((data[0], data[1], data[2]))
    # data = (data - np.mean(data)) / np.std(data)
    # default_data = np.ones(len(data))
    # length = np.size(data,1)
    dictionary = {'Delta-LRratio' : data[freq_bands[0,0]:freq_bands[0,1]+1],
                  'Theta-LRratio' : data[freq_bands[1,0]:freq_bands[1,1]+1],
                  'Alpha-LRratio' : data[freq_bands[2,0]:freq_bands[2,1]+1],
                  'Beta-LRratio' : data[freq_bands[3,0]:freq_bands[3,1]+1],
                  'Gamma1-LRratio' : data[freq_bands[4,0]:freq_bands[4,1]+1],
                  'Gamma2-LRratio' : data[freq_bands[5,0]:freq_bands[5,1]+1],
                  'Gamma3-LRratio' : data[freq_bands[6,0]:freq_bands[6,1]+1]
                  
                  # 'Delta-default' : default_data[freq_bands[0,0]:freq_bands[0,1]+1],
                  # 'Theta-default' : default_data[freq_bands[1,0]:freq_bands[1,1]+1],
                  # 'Alpha-default' : default_data[freq_bands[2,0]:freq_bands[2,1]+1],
                  # 'Beta-default' : default_data[freq_bands[3,0]:freq_bands[3,1]+1],
                  # 'Gamma1-default' : default_data[freq_bands[4,0]:freq_bands[4,1]+1],
                  # 'Gamma2-default' : default_data[freq_bands[5,0]:freq_bands[5,1]+1],
                  # 'Gamma3-default' : default_data[freq_bands[6,0]:freq_bands[6,1]+1],
                       }
    
    df = pd.DataFrame.from_dict(dictionary, orient='index')
    df=df.transpose()
    
    df=df.fillna(method ='bfill')
    
    df.to_csv(saving_directory + '/' + explanation, index=False)
#%% POT Data
#%% CSD
fild_epochs_icacsd, fild_epochslist_icacsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/Manuel Cleaned Epochs/ICA Yes, SSP No, CSD Yes/fild_cleanedepochs_icacsd','rb'))
jarrot_epochs_icasspcsd, jarrot_events, jarrot_epochlist_icasspcsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/Manuel Cleaned Epochs/ICA Yes, SSP Yes, CSD Yes/jarrot_cleanedepochs_icasspcsd','rb'))
sergio_epochs_icacsd, sergio_events, sergio_epochlist_icacsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/Manuel Cleaned Epochs/ICA Yes, SSP No, CSD Yes/sergio_cleanedepochs_icacsd','rb'))
lucireta_epochs_icasspcsd, lucireta_epochlist_icasspcsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/Manuel Cleaned Epochs/ICA Yes, SSP Yes, CSD Yes/lucireta_cleanedepochs_icasspcsd','rb'))
#%% NO-zscore
fild_epochs_ica_noz, fild_epochslist_ica_noz = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/Manuel Cleaned Epochs/ICA Yes, SSP No, CSD No/No-zscore/fild_cleanedepochs_ica','rb'))
fild_epochs_icacsd_noz, fild_epochslist_icacsd_noz = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/Manuel Cleaned Epochs/ICA Yes, SSP No, CSD Yes/No-zscore/fild_cleanedepochs_icacsd','rb'))
lucireta_epochs_icassp_noz, lucireta_epochlist_icassp_noz = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/Manuel Cleaned Epochs/ICA Yes, SSP Yes, CSD No/No-zscore/lucireta_cleanedepochs_icassp','rb'))
lucireta_epochs_icasspcsd_noz, lucireta_epochlist_icasspcsd_noz = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/Manuel Cleaned Epochs/ICA Yes, SSP Yes, CSD Yes/No-zscore/lucireta_cleanedepochs_icasspcsd','rb'))
#%% Load POT Data with Unit Based Normalization ===========
fild_epochs_ica, _, fild_epochslist_ica = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD No (Unit Factor Normalized)/fild_epochs_events_nobaseline','rb'))
sergio_epochs_ica, _, sergio_epochslist_ica = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD No (Unit Factor Normalized)/sergio_epochs_events_nobaseline','rb'))
jarrot_epochs_ica, _, jarrot_epochslist_ica = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD No (Unit Factor Normalized)/jarrot_epochs_events_nobaseline','rb'))
# lucireta_epochs_ica, lucireta_epochlist_ica = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/Manuel Cleaned Epochs/ICA Yes, SSP No, CSD No (Unit Factor Normalized)/lucireta_cleanedepochs_icassp','rb'))
erlacher_epochs_nothing, _, erlacher_epochlist_nothing = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/Manuel Cleaned Epochs/ICA No, SSP No, CSD No (Unit Factor Normalized)/o_kniebeugen_epochs_events_nobaseline','rb'))
#%% Load CSD Data with Unit Based Normalization ===========
fild_epochs_icacsd, fild_epochslist_icacsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes (Unit Factor Normalized)/fild_epochs_events_nobaseline','rb'))
sergio_epochs_icacsd, _, sergio_epochslist_icacsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes (Unit Factor Normalized)/sergio_epochs_events_nobaseline','rb'))
jarrot_epochs_icacsd, _, jarrot_epochslist_icacsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes (Unit Factor Normalized)/jarrot_epochs_events_nobaseline','rb'))
lucireta_epochs_icacsd, lucireta_epochlist_icacsd = pickle.load(open('/home/caghangir/Desktop/PhD/Lucid Dream EEG/Extracted Dataset/Epochs/ICA Yes, SSP No, CSD Yes (Unit Factor Normalized)/lucireta_cleanedepochs_icassp','rb'))


#%% ======== Periodograms [%] ==============

epochs_pot, epochs_csd = lucireta_epochs_icassp_noz, lucireta_epochs_icasspcsd_noz

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/CleanedEpochs/POT vs CSD comparison/Periodograms [%]'
saving_directory_statisticaltest = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/CleanedEpochs/POT vs CSD comparison/Statistical Tests'

fig, axs = plt.subplots(2, 1)

periodograms_pot, freqs = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=epochs_pot, events=epochs_pot.events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Periodogram [%] of POT, Lucid vs. REM vs. Wake', saving_directory=None, custom_ax=axs[0])

periodograms_csd, freqs = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=epochs_csd, events=epochs_csd.events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Periodogram [%] of CSD, Lucid vs. REM vs. Wake', saving_directory=None, custom_ax=axs[1])
    
fig.suptitle('Periodogram [%] of Jarrod`s Dataset \nBetween Scalp Potential (POT) and Current Source Density (CSD)', size=30)

#==== Statistical Dataset Generation =====
periodograms_pot = np.round(periodograms_pot, 4)
# LD_dataset_creation_forRMA(data=periodograms_pot, freqs=freqs, saving_directory=saving_directory_statisticaltest, \
#                                   explanation='JASP Connector.csv')
# LD_dataset_creation_forRMA(data=periodograms_csd, freqs=freqs, saving_directory=saving_directory_statisticaltest, \
#                                   explanation='JASP Connector.csv')
#==== Statistical Dataset Generation =====

# ===Save Figure ====
# if(saving_directory is not None):
#     chetto_EEG.save_figure(saving_directory, explanation='Lucireta, POT vs. CSD, Lucid vs. REM vs. Wake', extra='Grand Avg', dpi=400)
#===Save Figure ====
#%% ================ Periodogram [%] Merge 6-channels =========

#==== POT =====
jarrot_picked = jarrot_epochs_icassp.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
sergio_picked = sergio_epochs_ica.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
lucireta_picked = lucireta_epochs_icassp.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
merged_pot_6channels = chetto_EEG.epoch_concatenation([fild_epochs_icacsd, sergio_picked, lucireta_picked])
# merged_pot_6channels.apply_baseline(baseline=(0,0.2))
# merged_pot_6channels._data = zscore(merged_pot_6channels._data, axis=2)
#==== POT =====

#==== CSD =====
jarrot_picked = jarrot_epochs_icasspcsd.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
sergio_picked = sergio_epochs_icacsd.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
lucireta_picked = lucireta_epochs_icasspcsd.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
merged_csd_6channels = chetto_EEG.epoch_concatenation([fild_epochs_icacsd, sergio_picked, lucireta_picked])
# merged_csd_6channels.apply_baseline(baseline=(0,0.2))
# merged_pot_6channels._data = zscore(merged_pot_6channels._data, axis=2)
#==== CSD =====

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/CleanedEpochs/POT vs CSD comparison/Periodograms [%]'
saving_directory_statisticaltest = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/CleanedEpochs/POT vs CSD comparison/Statistical Tests'

fig, axs = plt.subplots(2, 1)

periodograms_pot, freqs = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=merged_pot_6channels, events=merged_pot_6channels.events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Periodogram [%] of POT, Lucid vs. REM vs. Wake', saving_directory=None, custom_ax=axs[0])
    
periodograms_csd, freqs = chetto_EEG.psd_spectrum_standardized_stage_comparison_plot(epochs=merged_csd_6channels, events=merged_csd_6channels.events, fmin=0.1, fmax=48, \
n_overlap=128, standardization=True, psd_type='multitaper', explanation='Periodogram [%] of CSD, Lucid vs. REM vs. Wake', saving_directory=None, custom_ax=axs[1])

fig.suptitle('Grand Average Periodogram [%] \nBetween Scalp Potential (POT) and Current Source Density (CSD)', size=30)

#==== Statistical Dataset Generation =====
LD_dataset_creation_forRMA(data=periodograms_pot, freqs=freqs, saving_directory=saving_directory_statisticaltest, \
                                 explanation='JASP Connector.csv') #GrandAVG_peridogram[%]_POT_dataset_RMA
# LD_dataset_creation_forRMA(data=periodograms_csd, freqs=freqs, saving_directory=saving_directory_statisticaltest, \
#                                   explanation='JASP Connector.csv') #GrandAVG_peridogram[%]_CSD_dataset_RMA
#==== Statistical Dataset Generation =====

# ===Save Figure ====
# if(saving_directory is not None):
#     chetto_EEG.save_figure(saving_directory, explanation='Grand Average, POT vs. CSD, Lucid vs. REM vs. Wake', extra='Grand Avg', dpi=400)
#===Save Figure ====

#%% ============== Lucid / REM Ratio ============
epochs_pot, epochs_csd = lucireta_epochs_icassp_noz, lucireta_epochs_icasspcsd_noz

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/CleanedEpochs/POT vs CSD comparison/PSD Lucid to Rem Ratio'
saving_directory_statisticaltest = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/CleanedEpochs/POT vs CSD comparison/Statistical Tests'

fig, axs = plt.subplots(2, 1)

_, grand_avg_LR_ratio_POT, freqs = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=epochs_pot, events=epochs_pot.events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='Lucid to REM Ratio of POT', smoothing=True, \
                                                     n_overlap=128, standardization=True, psd_type='multitaper', custom_ax=axs[0],\
                                                     ylim=[0.8, 1.2])

_, grand_avg_LR_ratio_CSD, freqs = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=epochs_csd, events=epochs_csd.events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='Lucid to REM Ratio of CSD', smoothing=True, \
                                                     n_overlap=128, standardization=True, psd_type='multitaper', custom_ax=axs[1],\
                                                     ylim=[0.8, 1.2])
    
fig.suptitle('Lucid/REM Ratio of Lucireta Dataset \nBetween Scalp Potential (POT) and Current Source Density (CSD)', size=30)

#==== Statistical Dataset Generation =====
# LD_dataset_creation_forRMA_LRratio(data=grand_avg_LR_ratio_POT, freqs=freqs, saving_directory=saving_directory_statisticaltest, \
#                                   explanation='JASP Connector.csv') #GrandAVG_peridogram[%]_POT_dataset_RMA
LD_dataset_creation_forRMA_LRratio(data=grand_avg_LR_ratio_CSD, freqs=freqs, saving_directory=saving_directory_statisticaltest, \
                                  explanation='JASP Connector.csv') #GrandAVG_peridogram[%]_CSD_dataset_RMA
#==== Statistical Dataset Generation =====

# ===Save Figure ====
# if(saving_directory is not None):
#     chetto_EEG.save_figure(saving_directory, explanation='Lucireta Lucid to REM Ratio, POT vs. CSD', extra='', dpi=400)
#===Save Figure ====
#%% ================ All Data Merge 6-channels =========

#==== POT =====
jarrot_picked = jarrot_epochs_icassp.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
sergio_picked = sergio_epochs_ica.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
lucireta_picked = lucireta_epochs_icassp_noz.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
merged_pot_6channels = chetto_EEG.epoch_concatenation([fild_epochs_ica_noz, sergio_picked, lucireta_picked])
merged_pot_6channels.apply_baseline(baseline=(0,0.2))
# merged_pot_6channels._data = zscore(merged_pot_6channels._data, axis=2)
#==== POT =====

#==== CSD =====
jarrot_picked = jarrot_epochs_icasspcsd.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
sergio_picked = sergio_epochs_icacsd.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
lucireta_picked = lucireta_epochs_icasspcsd_noz.pick_channels(['F3', 'F4', 'C3', 'C4', 'O1', 'O2'])
merged_csd_6channels = chetto_EEG.epoch_concatenation([fild_epochs_icacsd_noz, sergio_picked, lucireta_picked])
merged_csd_6channels.apply_baseline(baseline=(0,0.2))
# merged_csd_6channels._data = zscore(merged_csd_6channels._data, axis=2)
#==== CSD =====
#%% ========== Lucid / REM Ratio of Merged Data ========
saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/CleanedEpochs/POT vs CSD comparison/PSD Lucid to Rem Ratio'

fig, axs = plt.subplots(2, 1)

_, grand_avg_LR_ratio_POT, freqs = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=merged_pot_6channels, events=merged_pot_6channels.events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='Lucid to REM Ratio of POT', smoothing=True, \
                                                     n_overlap=128, standardization=True, psd_type='multitaper', custom_ax=axs[0],\
                                                     ylim=[0.9, 1.1])

_, grand_avg_LR_ratio_CSD, freqs = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=merged_csd_6channels, events=merged_csd_6channels.events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='Lucid to REM Ratio of CSD', smoothing=True, \
                                                     n_overlap=128, standardization=True, psd_type='multitaper', custom_ax=axs[1],\
                                                     ylim=[0.9, 1.1])
fig.suptitle('Grand Average Lucid/REM Ratio of PSD \nBetween Scalp Potential (POT) and Current Source Density (CSD)', size=30)
    
#==== Statistical Dataset Generation =====
# LD_dataset_creation_forRMA(data=periodograms_pot, freqs=freqs, saving_directory=saving_directory_statisticaltest, \
#                                  explanation='JASP Connector.csv') #GrandAVG_peridogram[%]_POT_dataset_RMA
# LD_dataset_creation_forRMA(data=periodograms_csd, freqs=freqs, saving_directory=saving_directory_statisticaltest, \
#                                   explanation='JASP Connector.csv') #GrandAVG_peridogram[%]_CSD_dataset_RMA
#==== Statistical Dataset Generation =====

# ===Save Figure ====
if(saving_directory is not None):
    chetto_EEG.save_figure(saving_directory, explanation='Grand Average, POT vs. CSD', extra='Grand Avg', dpi=400)
#===Save Figure ====
#%% ==================== Lucid / REM Ratio based on brain-region =====================
epochs_pot, epochs_csd = jarrot_epochs_icassp, jarrot_epochs_icasspcsd
epochs_csd.picks = np.array(epochs_csd._channel_type_idx['csd'])
datasetname = 'Jarrot'

# regions = ['Frontal', 'Fronto-lateral', 'Central', 'Temporal', 'Parietal', 'Occipital']
regions = ['Frontal', 'Central', 'Occipital']
# regions = ['Left anterior inferior', 'Frontal pole', 'Right anterior inferior', 'Left anterior superior', 'Right anterior superior',\
#            'Left posterior inferior', 'Posterior medial', 'Right posterior superior', 'Left posterior inferior', 'Right posterior inferior']
# EEG_channels = [['Fp1','Fp2','F3','F4','Fz'],['F7','F8'],['C3','C4','Cz'],['T3','T4','T5','T6'],['P3','P4','Pz'],['O1','O2']]
EEG_channels = [['F3','F4'],['C3','C4'],['O1','O2']]
# EEG_channels = [['Fp1','AF7','AF3','AFp1','AFF1h','AFF5h']]

color = ['#d74b1e', '#d7cc1e', '#6cd71e', '#15c8ce', '#2567e1', '#8625e1', '#bbbd16', '#ef67f6', '#898b9c', '#a3ca65']

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/CleanedEpochs/POT vs CSD comparison/PSD Lucid to REM Ratio_BrainRegions'
saving_directory_statisticaltest = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/CleanedEpochs/POT vs CSD comparison/Statistical Tests'

fig, axs = plt.subplots(2, 1)

grand_avg_LR_ratio_ROIs_POT = np.zeros((len(regions), 191))
for i in range(len(regions)):
    _, grand_avg_LR_ratio_POT, freqs = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=epochs_pot.copy().pick_channels(EEG_channels[i]), events=epochs_pot.events, fmin=0.1, fmax=48, \
                                                         saving_directory=None, explanation='Lucid to REM Ratio of POT', smoothing=True, \
                                                         n_overlap=128, standardization=True, psd_type='multitaper', custom_ax=axs[0],\
                                                         ylim=[0.5, 1.5], channelbychannel=False, label=regions[i], color=color[i], linestyle='solid')
    grand_avg_LR_ratio_ROIs_POT[i] = grand_avg_LR_ratio_POT

grand_avg_LR_ratio_ROIs_CSD = np.zeros((len(regions), 191))
for i in range(len(regions)):
    _, grand_avg_LR_ratio_CSD, freqs = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=epochs_csd.copy().pick_channels(EEG_channels[i]), events=epochs_csd.events, fmin=0.1, fmax=48, \
                                                         saving_directory=None, explanation='Lucid to REM Ratio of CSD', smoothing=True, \
                                                         n_overlap=128, standardization=True, psd_type='multitaper', custom_ax=axs[1],\
                                                         ylim=[0.5, 1.5], channelbychannel=False, label=regions[i], color=color[i], linestyle='solid')
    grand_avg_LR_ratio_ROIs_CSD[i] = grand_avg_LR_ratio_CSD
        
# fig.suptitle(datasetname + ' Dataset, Region of Interests (ROIs) Lucid/REM Ratio of PSD \nBetween Frontal, Central and Occipital', size=30)
fig.suptitle(datasetname + ', Region of Interests (ROIs) Lucid/REM Ratio of PSD \nBetween Frontal, Central and Occipital', size=30)

#==== Statistical Dataset Generation =====
# LD_dataset_creation_ROIs_3regions_forANOVA(data=grand_avg_LR_ratio_ROIs_POT, freqs=freqs, saving_directory=saving_directory_statisticaltest, \
#                                   explanation='JASP Connector.csv') #GrandAVG_peridogram[%]_POT_dataset_RMA
LD_dataset_creation_ROIs_3regions_forANOVA(data=grand_avg_LR_ratio_ROIs_CSD, freqs=freqs, saving_directory=saving_directory_statisticaltest, \
                                  explanation='JASP Connector.csv') #GrandAVG_peridogram[%]_CSD_dataset_RMA
#==== Statistical Dataset Generation =====

# ===Save Figure ====
# if(saving_directory is not None):
#     chetto_EEG.save_figure(saving_directory, explanation= datasetname + ', ROIs POT vs. CSD', extra='', dpi=400)
#===Save Figure ====
#%%===== Lucid / REM Topomap ====
epochs_pot, epochs_csd = fild_epochs_ica_noz, fild_epochs_icacsd_noz
datasetname = 'FILD'
vmin_pot, vmax_pot = 0.7, 1.3
vmin_csd, vmax_csd = 0.9, 1.1

#============ POT =================
lucid_rem_ratio, _,frequencies = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=epochs_pot, events=epochs_pot.events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='Lucid to REM Ratio of POT Frontal vs. Central vs. Occipital Regions', smoothing=True, \
                                                     n_overlap=128, standardization=True, psd_type='multitaper', \
                                                     ylim=[0.95, 1.08], channelbychannel=False, plot=False)

freq_bands = np.zeros((7,2)).astype(int)
freq_bands[0,0], freq_bands[0,1] = min(np.argwhere(frequencies >= 0))[0], max(np.argwhere(frequencies < 4))[0]
freq_bands[1,0], freq_bands[1,1] = min(np.argwhere(frequencies >= 4))[0], max(np.argwhere(frequencies < 8))[0] 
freq_bands[2,0], freq_bands[2,1] = min(np.argwhere(frequencies >= 8))[0], max(np.argwhere(frequencies < 12))[0]
freq_bands[3,0], freq_bands[3,1] = min(np.argwhere(frequencies >= 12))[0], max(np.argwhere(frequencies < 30))[0]
freq_bands[4,0], freq_bands[4,1] = min(np.argwhere(frequencies >= 30))[0], max(np.argwhere(frequencies < 40))[0]
freq_bands[5,0], freq_bands[5,1] = min(np.argwhere(frequencies >= 40))[0], max(np.argwhere(frequencies < 45))[0]
freq_bands[6,0], freq_bands[6,1] = min(np.argwhere(frequencies >= 45))[0], max(np.argwhere(frequencies < 48))[0]

    
evoked = epochs_pot.average()
evoked_delta = evoked.copy()
evoked_delta._data = lucid_rem_ratio
evoked_delta._data = evoked_delta._data
evoked_delta.times = np.array(range(0,191))

fig, ax = plt.subplots(nrows=2, ncols=8)
evoked_delta.plot_topomap(times=freq_bands[:,0], colorbar=True, cmap='Spectral_r', vmin=vmin_pot, \
                          vmax=vmax_pot, scalings=1, units='Lucid/REM Ratio', outlines='skirt', size=4, axes=ax[0])
figure = plt.gcf()
fig.text(0.35,0.9,'Lucid/REM Ratio of POT Among Frequency-Bands', fontsize=25, weight="bold")
fig.suptitle(datasetname + ' Dataset, Lucid/REM Ratio of POT Among Frequency-Bands', size=30)
ax[0,0].set_title('Delta (0-4 Hz)', size=20)
ax[0,1].set_title('Theta (4-8 Hz)', size=20)
ax[0,2].set_title('Alpha (8-12 Hz)', size=20)
ax[0,3].set_title('Beta (12-30 Hz)', size=20)
ax[0,4].set_title('Gamma-1 (30-40 Hz)', size=20)
ax[0,5].set_title('Gamma-2 (40-45 Hz)', size=20)
ax[0,6].set_title('Gamma-3 (45+ Hz)', size=20)
ax[0,7].set_title('Lucid/REM Ratio', size=20)
ax[0,7].tick_params(labelsize=15)
#============ POT =================

#============ CSD =================
lucid_rem_ratio, _,frequencies = chetto_EEG.psd_spectrum_standardized_lucid_REM_ratio(epochs=epochs_csd, events=epochs_csd.events, fmin=0.1, fmax=48, \
                                                     saving_directory=None, explanation='Lucid to REM Ratio of POT Frontal vs. Central vs. Occipital Regions', smoothing=True, \
                                                     n_overlap=128, standardization=True, psd_type='multitaper', \
                                                     ylim=[0.95, 1.08], channelbychannel=False, plot=False)

evoked = epochs_csd.average()
evoked_delta = evoked.copy()
evoked_delta._data = lucid_rem_ratio
evoked_delta._data = evoked_delta._data
evoked_delta.times = np.array(range(0,191))

evoked_delta.plot_topomap(times=freq_bands[:,0], colorbar=True, cmap='Spectral_r', vmin=vmin_csd, \
                          vmax=vmax_csd, scalings=1, units='Lucid/REM Ratio', outlines='skirt', size=4, axes=ax[1])
figure = plt.gcf()
fig.text(0.35,0.4,'Lucid/REM Ratio of CSD Among Frequency-Bands', fontsize=25, weight="bold")
ax[1,0].set_title('Delta (0-4 Hz)', size=20)
ax[1,1].set_title('Theta (4-8 Hz)', size=20)
ax[1,2].set_title('Alpha (8-12 Hz)', size=20)
ax[1,3].set_title('Beta (12-30 Hz)', size=20)
ax[1,4].set_title('Gamma-1 (30-40 Hz)', size=20)
ax[1,5].set_title('Gamma-2 (40-45 Hz)', size=20)
ax[1,6].set_title('Gamma-3 (45+ Hz)', size=20)
ax[1,7].set_title('Lucid/REM Ratio', size=20)
ax[1,7].tick_params(labelsize=15)
#============ CSD =================

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/CleanedEpochs/POT vs CSD comparison/PSD Lucid to REM Ratio_BrainRegions/TopoMaps'
# ===Save Figure ====
if(saving_directory is not None):
    chetto_EEG.save_figure(saving_directory, explanation = datasetname + ', Topomap POT vs. CSD', extra='', dpi=400)
#===Save Figure ====
#%% ============== Sergio & LUCIRETA 40 Hz TopoMap =============
epochs_pot, epochs_csd = sergio_epochs_ica, sergio_epochs_icacsd

# evoked = lucireta_epochs_icasspcsd.average()
# evoked.plot()
fig, ax = plt.subplots(nrows=2, ncols=3)
ax = np.resize(ax, (1,2,3)) #to make each ax element also an array
epochs_pot['REM'].plot_psd_topomap(ch_type='eeg', dB=True, proj=True, bands=[(36,45, 'γ-40 Hz REM')], cmap='Spectral_r', axes=ax[:,0,0])
epochs_pot['Lucid'].plot_psd_topomap(ch_type='eeg', dB=True, proj=True, bands=[(36,45, 'γ-40 Hz Lucid')], cmap='Spectral_r', axes=ax[:,0,1])
epochs_pot['Wake'].plot_psd_topomap(ch_type='eeg', dB=True, proj=True, bands=[(36,45, 'γ-40 Hz Wake')], cmap='Spectral_r', axes=ax[:,0,2])

fig.suptitle('Sergio Dataset, PSD Topomap of γ-40 Hz (36-45 Hz) \n REM vs. Lucid vs. Wake', size=30)

fig.text(0.45,0.88,'γ-40 Hz POT', fontsize=25, weight="bold")
ax[0,0,0].set_title('γ-40 Hz REM', size=20) 
ax[0,0,1].set_title('γ-40 Hz Lucid', size=20) 
ax[0,0,2].set_title('γ-40 Hz Wake', size=20) 

epochs_csd['REM'].plot_psd_topomap(ch_type='csd', dB=True, proj=True, bands=[(36,45, 'γ-40 Hz REM')], cmap='Spectral_r', axes=ax[:,1,0])
epochs_csd['Lucid'].plot_psd_topomap(ch_type='csd', dB=True, proj=True, bands=[(36,45, 'γ-40 Hz Lucid')], cmap='Spectral_r', axes=ax[:,1,1])
epochs_csd['Wake'].plot_psd_topomap(ch_type='csd', dB=True, proj=True, bands=[(36,45, 'γ-40 Hz Wake')], cmap='Spectral_r', axes=ax[:,1,2])

fig.text(0.45,0.45,'γ-40 Hz CSD', fontsize=25, weight="bold")
ax[0,1,0].set_title('γ-40 Hz REM', size=20) 
# ax[0,1,0].tick_params(labelsize=15)
ax[0,1,1].set_title('γ-40 Hz Lucid', size=20) 
ax[0,1,2].set_title('γ-40 Hz Wake', size=20) 

plt.subplots_adjust(top=0.882, wspace=0.785, hspace=0)

# plt.subplot_tool()
# plt.show()

saving_directory = '/home/caghangir/Desktop/PhD/Lucid Dream EEG/Progress_Cagatay/26-12-2020/CleanedEpochs/POT vs CSD comparison/40 Hz Band'
# ===Save Figure ====
if(saving_directory is not None):
    chetto_EEG.save_figure(saving_directory, explanation= 'Sergio, 40Hz Topomap REM vs. Lucid vs. Wake', extra='', dpi=400)
#===Save Figure ====

# plt.subplot_tool()

#%% =============== Area 51 ===========
fig, ax = plt.subplots(nrows=2, ncols=7)
evoked_delta.plot_topomap(times=freq_bands[:,0], colorbar=True, cmap='Spectral_r', vmin=np.min(evoked_delta._data), \
                          vmax=np.max(evoked_delta._data), scalings=2, units='Lucid/REM Ratio', outlines='skirt', size=4, axes=ax[0], title='xx')
# figure = plt.gcf()
fig.suptitle('LUCIRETA Dataset, Lucid/REM Ratio of POT Among Frequency-Bands', size=30)
ax[0,0].set_title('Delta (0-4 Hz)', size=20)
ax[0,1].set_title('Theta (4-8 Hz)', size=20)
ax[0,2].set_title('Alpha (8-12 Hz)', size=20)
ax[0,3].set_title('Beta (12-30 Hz)', size=20)
ax[0,4].set_title('Gamma (30-45 Hz)', size=20)
ax[0,5].set_title('Higher Gamma (30-45 Hz)', size=20)
ax[0,6].set_title('Lucid/REM Ratio', size=20)
ax[0,6].tick_params(labelsize=15)

fig.text(0.35,0.9,'Lucid/REM Ratio of POT Among Frequency-Bands', fontsize=25, weight="bold")

evoked_delta.plot_topomap(times=freq_bands[:,0], colorbar=True, cmap='Spectral_r', vmin=np.min(evoked_delta._data), \
                          vmax=np.max(evoked_delta._data), scalings=1, units='Lucid/REM Ratio', outlines='skirt', size=4, axes=ax[1])
# figure = plt.gcf()
fig.suptitle('LUCIRETA Dataset, Lucid/REM Ratio of POT Among Frequency-Bands', size=30)
ax[1,0].set_title('Delta (0-4 Hz)', size=20)
ax[1,1].set_title('Theta (4-8 Hz)', size=20)
ax[1,2].set_title('Alpha (8-12 Hz)', size=20)
ax[1,3].set_title('Beta (12-30 Hz)', size=20)
ax[1,4].set_title('Gamma (30-45 Hz)', size=20)
ax[1,5].set_title('Higher Gamma (30-45 Hz)', size=20)
ax[1,6].set_title('Lucid/REM Ratio', size=20)
ax[1,6].tick_params(labelsize=15)

fig.text(0.35,0.4,'Lucid/REM Ratio of CSD Among Frequency-Bands', fontsize=25, weight="bold")
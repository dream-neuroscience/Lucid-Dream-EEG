# Lucid-Dream-EEG
The repository regarding to Lucid Dream EEG Project

************************ Most important Scripts ******************************
1) lucid_dream_EEG_analysis_epoching.py : It creates epochs from folder paths of all datasets

3) lucid_dream_EEG_analysis_spectralanalysis_comparative.py : It does comparative spectral analysis between Scalp power and CSD.

************************** Pipeline Functions **********************************

Pipeline functions are here to make our life easier. They are functions who use functions in ordered form to generate high-end goals at one run. 
Main pipelines:
* eeg_epoching_pipeline_of_given_folder(): It gets folder path, time interval for each file and couple other inputs to extract epochs of MNE structure.
  Here you can find examples in this script; lucid_dream_EEG_analysis_epoching.py() 
  
  Epoching pipeline steps:
  - Read edf & brainvision file
  - Drop channel if needed
  - Sensory information update:
    1) Rename channels
    2) Pick channels
    3) Set channel types
    4) Set montage layout
  - Preprocessing
    1) Unit normalization
    2) Resampling
    3) Notch filtering
    4) Band-pass filtering
    5) ICA
    6) SSP (Signal space projection)
    7) CSD (Common source density)
  - Epoching based on events (time intervals and specific window size with padding size)
  - Concatenation of epochs for each file (at the end you have giant epoch array contains epochs from all the files in the same folder).
  
* eeg_neighbourhoodinterval_of_given_folder() : It gets again folder path to generate neighbourhood frame of given Lucid dream time-interval.

# Lucid-Dream-EEG
The repository regarding to Lucid Dream EEG Project

There is a library I created which has multiple functions for different purposes. It is called extremeEEGSignalAnalyzer. The scripts in this project mainly use
this library.

Features about the library :
* Mainly uses MNE background for EEG data analysis including preprocess, temporal & spectral analysis, some other analysis and prospected source level analysis
  in the future.
* Hand-crafted EEG feature extraction:
  - Petriosian Fractal dimension (PFD)
  - Hjorth fractal dimension (HFD)
  - Hjorth motibility and complexity of a time-series
  - Hurst exponent
  - Detrended fluctuation analysis (DFA)
  - Signal energy of frame
  - Spectral Entropy
  - Entropy of energy
  - RMS value
  - Linear prediction coefficient
* Spectral & temporal feature fusion in short, medium and long term windows
* Brainwave extraction
* Statistical features:
  - kurtosis, skewness, mean, median, std
* Z-score normalization
* Wavelet decomposition
* First and second difference mean and max
* Variance and Mean of Vertex to Vertex Slope
* Hypnogram Comparator
* Automatic EDF decomposer (extract basic information of multiple EDF files into the excel file)
* Confusion Matrix plot
* EEG cutter and saver (inputs are time intervals and outputs are cutted EEG frame images)
* Multi EEG files cutter and saver (inputs are time intervals of multiple EEG files and outputs are cutted EEG frame images)
* Spectrogram generation
* Read edf files
* Read brainvision files
* Event & Epoching of EEG files (edf or brainvision)
* Sensor location update
* ICA:
  - By default delete 1st ICA component
  - Auto part finds bads of EOG and ECG if there are these type of channels in your setup
* Preprocessing (Step-by-step in details of pipeline functions)
* Epoch concatenation
* Factor statistics

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


****************************** MNE + Python Combination Functions ******************************
* PSD Standardized Stage Comparison Plot (Voss method)
* PSD Standardized Lucid / REM ratio


****************************** Spectral Connectivity Analysis ************************************
* Spectral coherence (averaged and non-averaged)
* Multiple coherence algorithms at once:
  - Power coherence
  - Imaginary coherence
  - Weighted Phase Lag Index Coherence
  
***************************** MNE PLot Functions ************************************
* plot epochs
* plot image
* plot evoked topomap
* Multitaper spectrogram plot
* Plot sensor locations
* Plot global field power
* Plot evoked response


*************************** Other functions ******************************** (hard to tell, they are weird :/ )

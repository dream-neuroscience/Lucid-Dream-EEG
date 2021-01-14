# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:03:40 2020

@author: caghangir
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
# %matplotlib qt #plot in new window

#%% ========= Hypnogram Plotting ===================
def plot_hyp(hyp, fileName, mark_REM = 'active'):
        
    stages = hyp
    #stages = np.row_stack((stages, stages[-1]))
    x      = np.arange(len(stages))
    
    # Change the order of classes: REM and wake on top
    x = []
    y = []
    for i in np.arange(len(stages)):
        s = stages[i]
        if s== 0 :  p = -0
        if s== 4 :  p = -1
        if s== 1 :  p = -2
        if s== 2 :  p = -3
        if s== 3 :  p = -4
        if i!=0:
            y.append(p)
            x.append(i-1)   
    y.append(p)
    x.append(i)
    

    #plt.figure(figsize = [20,14])
    plt.step(x, y, where='post')
    plt.yticks([0,-1,-2,-3,-4], ['W','REM', 'N1', 'N2', 'SWS'])
    plt.ylabel('sleep stage')
    plt.xlabel('# epoch')
    plt.title('Hypnogram')
    plt.rcParams.update({'font.size': 20})
    
    # Mark REM epochs
    if mark_REM == 'active':
        rem = [i for i,j in enumerate(hyp) if (hyp[i]==4)]
        for i in np.arange(len(rem)) -1:
            if rem[i+1] - rem[i] == 1:
                plt.plot([rem[i], rem[i+1]], [-1,-1] , linewidth = 5, color = 'red')
            elif rem[i] - rem[i-1] == 1:
                plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
    
            elif ((rem[i+1] - rem[i] != 1) and (rem[i] - rem[i-1] != 1)):
                plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
                
    plt.savefig('Hypnogram :' + fileName, pad_inches=0, dpi=800)
#%% ======== Reading Data Info ============
os.chdir('C:\\Users\\caghangir\\Desktop\\PhD\\Research\\Lucid Dream EEG\\Total Data\\2_Christian_Tausch')

files = list()
for file in os.listdir("O_Kniebeugen"):
    if file.endswith(".sl"):
        files.append(file)

for file in files:
    temp_file = open('O_Kniebeugen/' + files[1],"r")     
    
    content = temp_file.read()
    totalStrings = list()
    lengths = list()
    tempString = ''
    tempLength = 0
    stringDetected = 0
    gapDetected = 0
    for i in range(len(content)):
        if(content[i] != ' '):
            tempString += str(content[i])
            stringDetected = 1
            if(gapDetected == 1): #only enter one time if you find beginning of the new string 
                lengths.append(tempLength)
                tempLength = 0
                gapDetected = 0
            
        elif(stringDetected == 1): #if you find end of string
            totalStrings.append(tempString)
            stringDetected = 0
            tempString = '' #reset the temp string
            tempLength += 1
            gapDetected = 1
        elif(stringDetected == 0): #inside the gap
            tempLength += 1
            gapDetected = 1
    
    totalStrings = np.array(totalStrings)
    lengths = np.array(lengths)
    uniques = np.unique(totalStrings)
    
    #==== Mapping Strings ========
    mappedStrings = np.empty(shape=[0])
    for i in range(len(totalStrings)-1):
        temp_index = np.argwhere(uniques == totalStrings[i])
        temp_gap_indexes = np.ones(lengths[i]+1) * temp_index #+1 for original temp_index
        mappedStrings = np.append(mappedStrings, temp_gap_indexes)
    #==== Mapping Strings ========
        
    #========= Findings Stages =======
    hypnogram = np.zeros(len(mappedStrings)) - 2
    
    stage1_indexes = np.argwhere((mappedStrings == 0) | (mappedStrings == 1))
    stage2_indexes = np.argwhere((mappedStrings == 2) | (mappedStrings == 3))
    stage3_indexes = np.argwhere((mappedStrings == 4) | (mappedStrings == 5) | (mappedStrings == 6) | (mappedStrings == 7))
    stage_rem_indexes = np.argwhere((mappedStrings == 12) | (mappedStrings == 13) | (mappedStrings == 14) | \
                                    (mappedStrings == 15) | (mappedStrings == 16) | (mappedStrings == 17) | \
                                    (mappedStrings == 18) | (mappedStrings == 19) | (mappedStrings == 20) | \
                                    (mappedStrings == 21) | (mappedStrings == 22) | (mappedStrings == 23))
    stage_awake = np.argwhere((mappedStrings == 24) | (mappedStrings == 26))
    noise = np.argwhere((mappedStrings == 10) | (mappedStrings == 11) | (mappedStrings == 25)) 
    
    hypnogram[stage1_indexes] = 1
    hypnogram[stage2_indexes] = 2
    hypnogram[stage3_indexes] = 3
    hypnogram[stage_rem_indexes] = 4
    hypnogram[stage_awake] = 0
    hypnogram[noise] = -1
    
    totalIndex = len(stage1_indexes) + len(stage2_indexes) + len(stage3_indexes) + len(stage_rem_indexes) + len(noise) 
    
    ''' '1' is stage 1, '2' is stage 2, '3' is stage 3, '4' is stage REM, '0' is stage awake, '-1' is head movement '''
    #========= Findings Stages =======
    
    #===== Save Hypnogram Figure ========
    plot_hyp(hypnogram, file, mark_REM = 'active')
    #===== Save Hypnogram Figure ========

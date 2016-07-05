# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:12:35 2016

@author: jbraga
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import csv

def load_trdataset_list():

    ltrdataset=[]    
    cr = csv.reader(open('../traditional_dataset/dataset.csv',"rb"))
    for row in cr:
        ltrdataset.append(row[0]) 
        
    return ltrdataset
    
def load_gt(gt_file, t):
    
    cr = csv.reader(open(gt_file,"rb"))
    onset=[]
    notes=[]
    for row in cr:
        onset.append(row[0]) 
        notes.append(row[1])
    onset = np.array(onset, 'float32')
    
    aux_vad_gt = np.empty([0,], 'int8')
    for note in notes:
        if note=='0':
            aux_vad_gt = np.r_[aux_vad_gt,0]
        else:
            aux_vad_gt = np.r_[aux_vad_gt,1]
    
    j=0
    vad_gt = np.empty([len(t),], 'int8')
    for i in range(1,len(onset)):
        while (j<len(t) and t[j]>=onset[i-1] and t[j]<=onset[i]):
            vad_gt[j]=aux_vad_gt[i-1]
            j=j+1  
    
    return vad_gt
    
def load_audio(audio_file):
    
    fs, audio = wav.read(audio_file)
#    audio = audio.astype('float64')
    t = np.arange(len(audio)) * float(1)/fs
    
    return audio, t, fs    


if __name__ == "__main__":  


    ltrdataset = load_trdataset_list()    

    fragment = ltrdataset[8]    
    
    audio_file = fragment + '_mono.wav'
    gt_file = fragment + '.csv'

    audio, t, fs = load_audio(audio_file)
    vad_gt = load_gt(gt_file, t)    


#%% SPEC & WAVE WITH GT
    
    frame_size = 1024
    op = frame_size/2
    
    f, t_S, Sxx = signal.spectrogram(audio, fs, window='hamming', nperseg=frame_size, 
                                     noverlap=op, nfft=None, detrend='constant',
                                     return_onesided=True, scaling='spectrum', axis=-1)
            
    plt.figure(figsize=(18,6))
    plt.subplot(3,1,(1,2))
    plt.pcolormesh(t_S, f, 20*np.log(Sxx))
    plt.axis('tight')
    plt.subplot(3,1,3)
    plt.plot(t, audio, color='black')
    plt.grid()
    plt.axis('tight')
    plt.plot(t, vad_gt*max(audio)/2, color='blue', lw=2)
    plt.fill_between(t, 0, vad_gt*max(audio)/2, facecolor='blue', alpha=0.2)
    plt.show()

#%% ESTIMATE VAD

    import envelope as tenv

    temporal_env, t_env, audio_abs = tenv.morph_close(audio, fs=fs, n=frame_size)
    
    plt.figure(figsize=(18,6))
    plt.plot(t, audio, color='black', alpha=0.5)
    plt.grid()
    plt.axis('tight')
    plt.plot(t,  max(audio_abs)*temporal_env/max(temporal_env), color='blue', lw=0.5)
    plt.fill_between(t, 0, max(audio_abs)*temporal_env/max(temporal_env), facecolor='blue', alpha=0.8)
    plt.show()


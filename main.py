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
import tradataset as td


if __name__ == "__main__":  

    ltrdataset = td.load_list()    

    fragment = ltrdataset[8]    
    
    audio_file = fragment + '_mono.wav'
    gt_file = fragment + '.csv'

    audio, t, fs = td.load_audio(audio_file)
    vad_gt = td.load_gt(gt_file, t)    


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
    plt.plot(t, audio, color='black', alpha=0.7)
    plt.grid()
    plt.axis('tight')
    plt.fill_between(t, -vad_gt*(2**12), vad_gt*(2**12), facecolor='blue', alpha=0.2)
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


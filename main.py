# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:12:35 2016

@author: jbraga
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import tradataset as td
import frequency_to_notation as pe 


if __name__ == "__main__":  

    ltrdataset = td.load_list()    

    fragment = ltrdataset[8]    
    
    audio_file = fragment + '_mono.wav'
    gt_file = fragment + '.csv'

    audio, t, fs = td.load_audio(audio_file)
    vad_gt, gt = td.load_gt(gt_file, t)    

#%% SPEC & WAVE WITH GT
    
    frame_size = 1024
    op = 0
    
    f, t_S, Sxx = signal.spectrogram(audio, fs, window='hamming', nperseg=frame_size, 
                                     noverlap=op, nfft=None, detrend='constant',
                                     return_onesided=True, scaling='spectrum', axis=-1)
            
    plt.figure(figsize=(18,6))
    plt.subplot(3,1,(1,2))
    plt.pcolormesh(t_S, f, 20*np.log(Sxx))
    plt.axis('tight')
    plt.subplot(3,1,3)
    plt.plot(t, audio, color='black', alpha=0.6)
    plt.grid()
    plt.axis('tight')
    plt.fill_between(t, -vad_gt*(2**12), vad_gt*(2**12), facecolor='yellow', alpha=0.3)
    plt.show()

#%% ESTIMATE VAD

    import envelope as tenv

    temporal_env, t_env, audio_abs = tenv.morph_close(audio, fs=fs, n=frame_size)
    
    plt.figure(figsize=(18,6))
    plt.plot(t, audio, color='black', alpha=0.5)
    plt.grid()
    plt.axis('tight')
    plt.plot(t,  max(audio_abs)*temporal_env/max(temporal_env), color='blue', lw=0.5)
    #plt.fill_between(t, 0, max(audio_abs)*temporal_env/max(temporal_env), facecolor='blue', alpha=0.8)
    plt.show()
    
#%% ESTIMATE PITCH

   
    
    hop = (frame_size-op)
    pitch_midi, timestamps = pe.pitch_extraction(audio, fs, frame_size, hop)

    fig = plt.figure(figsize=(18,6))                                                               
    ax = fig.add_subplot(3,1,(1,2))                                                      
    
    yticks_major = [ 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 89, 91, 93, 95, 96 ]
    yticks_minor = [ 61, 63, 66, 68, 70, 73, 75, 78, 80, 82, 85, 87, 90, 92, 94 ]
#    yticks_labels = ['B3', 'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5', 'C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6', 'C7']                         
                                             
    ax.set_yticks(yticks_major)                                                       
#    ax.set_ytickslabels(yticks_labels)       
    ax.set_yticks(yticks_minor, minor=True)
                                    
    plt.ylim(58, 96) #flute register in midi   
    plt.xlim(0, t[-1])
    ax.grid(b=True, which='major', color='black', axis='y', linestyle='-', alpha=0.3)
    ax.grid(b=True, which='minor', color='black', axis='y', linestyle='-', alpha=1)    
    
    plt.plot(timestamps , pitch_midi,'.-',color='red', lw=0.3)
    plt.fill_between(t, gt-0.5, gt+0.5, facecolor='yellow', label='notes', alpha=0.6)

    plt.title("Melody")
    plt.ylabel("Pitch (Midi)")
    
    plt.subplot(3,1,3)
    plt.plot(t, audio, color='black', alpha=0.5)
    plt.grid()
    plt.axis('tight')
    plt.fill_between(t, -vad_gt*(2**12), vad_gt*(2**12), facecolor='cyan', alpha=0.6)    
    plt.title("WaveForm + Activity Detection")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")    
    plt.show()
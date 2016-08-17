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
import melosynth as ms
import librosa as lr
from dtw import dtw

def moving_average(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == "__main__":  

    ltrdataset = td.load_list()    

    fragment = ltrdataset[12]    
    
    audio_file = fragment + '_mono.wav'
    gt_file = fragment + '.csv'
    score_file = fragment + '.xml'

    audio, t, fs = td.load_audio(audio_file)
    vad_gt, gt, onset_gt = td.load_gt(gt_file, t)    

    frame_size = 1024
    op = 0    

#%% VOICE ACTIVITY DETECTION

    import envelope as tenv
    temporal_env, t_env, audio_abs = tenv.morph_close(audio, fs=fs, n=frame_size)
    
#%% PITCH EXTRACTION
    
    hop = (frame_size-op)
    pitch_midi, timestamps = pe.pitch_extraction(audio, fs, frame_size, hop)    
    
#%% NICE PLOTTING PITCH EXTRACTION VS. MUSIC ANNOTATIONS
    
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
    
    #plt.plot(timestamps, pitch_midi,'.-',color='red', lw=0.3)
    plt.plot(timestamps, pitch_midi,'.-',color='green', lw=0.3)
    plt.fill_between(timestamps, gt-0.5, gt+0.5, facecolor='yellow', label='notes', alpha=0.6)

#    for xc in limits[0]:
#        plt.axvline(timestamps[xc], color='black', linestyle='--', alpha=0.3)  

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
    
#%% DTW PITCH EXTRACTION VS. MUSIC ANNOTATIONS  

    x=pitch_midi.astype('float64')
    x=x.reshape(-1, 1)
    y=gt.reshape(-1, 1)
    dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
    print 'Minimum distance found:', dist
    plt.figure()    
    plt.imshow(acc.T, origin='lower', cmap='gray', interpolation='nearest')
    plt.pcolormesh(acc.T)
    plt.plot(path[0], path[1], 'w')
    plt.xlim((-0.5, acc.shape[0]-0.5))
    plt.ylim((-0.5, acc.shape[1]-0.5))

#%% PITCH EXTRACTION and GT .CSV EXPORT FOR MIR_EVAL
    
#    np.savetxt(fragment + "_melody_extraction.csv", np.c_[timestamps, filtrated_pitch.astype('int16')], delimiter=" ",fmt='%.14f')
#    np.savetxt(fragment + "_melody_gt.csv", np.c_[timestamps, gt.astype('int16')], delimiter="      ",fmt='%.14f')

#%% PITCH EXTRACTION AND GT SYNTHESIS FOR AUDITIVE CHECK
    
#    ms.melosynth_pitch(lr.midi_to_hz(filtrated_pitch), (fragment + '_extractionsynth.wav'), fs=44100, nHarmonics=1, square=True, useneg=False) 
#    ms.melosynth_pitch(lr.midi_to_hz(gt), (fragment + '_gtsynth.wav'), fs=44100, nHarmonics=1, square=True, useneg=False) 
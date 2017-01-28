# -*- coding: utf-8 -*-

from dtw import dtw
import frequency_to_notation as pe
import tradataset as td
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':

    ltrdataset = td.load_list()    

    fragment = ltrdataset[7]    
    
    audio_file = fragment + '_mono.wav'
    gt_file = fragment + '.csv'
    score_file = fragment + '.xml'       

    frame_size = 1024
    hop=frame_size
    
    audio, t, fs = td.load_audio(audio_file)
    vad_gt, gt, onset_gt, melo = td.load_gt(gt_file, t, frame=frame_size)        
    score, thenotes = td.load_score(score_file)
    
    #DTW GT VS. SYMBOLIC SCORE       
    proportion = len(gt)/len(score)
    if proportion>1:    
        score_aux=np.empty([proportion*len(score),])
        for i in range(0,len(score)):
            for j in range(proportion*i,proportion*(i+1)):        
                score_aux[j]=score[i]
    else:
        score_aux=score

    timestamps=np.arange(len(gt)) * float(frame_size)/fs
            
    t_score = np.arange(len(score_aux)) * (timestamps[-1]/len(score_aux)) 

    x=gt.astype('float64')
    x=x.reshape(-1, 1)
    y=score_aux.reshape(-1, 1)
    dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
    print 'Minimum distance found:', dist

#%%

    fig = plt.figure(figsize=(18,6))      
    plt.subplot(5,1,(4,5))    
    plt.imshow(acc.T, origin='lower', cmap='gray', interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.xlim((-0.5, acc.shape[0]-0.5))
    plt.ylim((-0.5, acc.shape[1]-0.5))

    t_dtw = np.arange(len(path[0])) * (timestamps[-1]/len(path[0])) 
    score_wrapped = score_aux[path[1]]    
    gt_wrapped = gt[path[0]]

    ax = fig.add_subplot(5,1,(1,3))
                                            
    yticks_major = [ 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86, 88, 89, 91, 93, 95, 96 ]
    yticks_minor = [ 61, 63, 66, 68, 70, 73, 75, 78, 80, 82, 85, 87, 90, 92, 94 ]
                                           
    ax.set_yticks(yticks_major)                                                           
    ax.set_yticks(yticks_minor, minor=True)
                                    
    plt.ylim(58, 96) #flute register in midi   
    plt.xlim(0, t_dtw[-1])
    ax.grid(b=True, which='major', color='black', axis='y', linestyle='-', alpha=0.3)
    ax.grid(b=True, which='minor', color='black', axis='y', linestyle='-', alpha=1)    
    plt.fill_between(t_dtw, score_wrapped-0.5, score_wrapped+0.5, facecolor='yellow', label='notes', alpha=0.6)
    plt.plot(t_dtw, gt_wrapped)
    
#%% EXPORT MIDI

    import audio_to_midi_melodia as atmm
    
    smooth=0
    minduration=0.1
    bpm=90
    
    # segment sequence into individual midi notes
    notes_gt = atmm.midi_to_notes(gt_wrapped, fs, hop, smooth, minduration)
    notes_score = atmm.midi_to_notes(score_wrapped, fs, hop, smooth, minduration)

    # save note sequence to a midi file
    print("Saving MIDI to disk...")
    atmm.save_midi('prueba_gt.mid', notes_gt, bpm)
    atmm.save_midi('prueba_score.mid', notes_score, bpm)

    print("Conversion complete.")
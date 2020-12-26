#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 00:33:37 2020

@author: krishna
"""

import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import pumpp
from scipy.signal import medfilt
from scipy.fft import dct, idct


audio_filepath ='/media/newhd/AntiSpoofing/DS_10283_3336/PA/ASVspoof2019_PA_dev/flac/PA_D_0024461.flac'
audio_filepath ='/media/newhd/AntiSpoofing/DS_10283_3336/PA/ASVspoof2019_PA_dev/flac/PA_D_0000071.flac'
audio_filepath = '/home/krishna/Krishna/paper_implementations/ASSERT/baseline/CQCC_v1.0/D18_1000001.wav'




def load_audio(audio_filepath):
    audio_data, fs = librosa.load(audio_filepath)
    return audio_data, fs

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=512):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def mgdf(audio_filepath):
    hop_length=160
    win_length=400
    rho = 0.4
    gamma = 0.9
    xn, fs = load_audio(audio_filepath)
    n_xn = xn*range(1,len(xn)+1)
    
    X = lin_spectogram_from_wav(xn, hop_length, win_length, n_fft=512)
    Y = lin_spectogram_from_wav(n_xn, hop_length, win_length, n_fft=512)
    Xr, Xi = np.real(X), np.imag(X)
    Yr, Yi = np.real(Y), np.imag(Y)
    magnitude,_ = librosa.magphase(X,1)
    S = np.square(np.abs(magnitude)) # powerspectrum
    dct_spec = dct(medfilt(S, 5));
    smooth_spec = np.abs(idct(x=dct_spec[:,:30],n=257))
    gd = (Xr*Yr + Xi*Yi)/np.power(smooth_spec+1e-05,rho)
    plt.imshow(gd)
    mgd = gd/np.abs(gd)*np.power(np.abs(gd),gamma)
    mgd = mgd/np.max(mgd)
    cep = np.log2(np.abs(mgd))
    return cep
    
    
    
def ifgram(audio_data):
    xn, fs = load_audio(audio_filepath)
    frequencies, D = librosa.ifgram(xn, sr=fs,n_fft=1024, win_length=400, hop_length=160)
    mag, _ = librosa.magphase(frequencies)
    

def trim_silence(audio, threshold=0.1, frame_length=2048):
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rms(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    
def CQCC(audio_filepath):
    p_cqt = pumpp.feature.CQT(name='cqt', sr=22050, hop_length=128)
    xn, fs = load_audio(audio_filepath)
    out = p_cqt.transform_audio(xn)
    C = out['mag']
    log_cqt = np.log2(np.power(np.abs(C),2))
    return log_cqt
    

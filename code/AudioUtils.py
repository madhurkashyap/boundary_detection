# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 06:06:41 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import librosa
import numpy as np
from sphfile import SPHFile
from python_speech_features import mfcc,logfbank

def read_sph(path):
    assert os.path.exists(path), "SPH file not readable"
    sph = SPHFile(path);
    return sph.format['sample_rate'], sph.content,

def extract_features(y,sr,n_mfcc=13,wlen=0.025,wstep=0.01,n_fft=2048,
                     roc=False,roa=False,logfb=False,method='librosa'):
    """
    fmax has been default to 8k allowing 4khz audio frequency samples
    """
    win_length = int(sr*wlen); hop_length = int(wstep*sr);
    n_mels = n_mfcc*2;
    if method=='psf':
        m = mfcc(y,samplerate=sr,winlen=wlen,winstep=wstep,numcep=n_mfcc)
        S = logfbank(y,sr,wlen,wstep);
        if roc or roa: d = librosa.feature.delta(m);
        if roa: dd = librosa.feature.delta(d);
        if roc: m = np.hstack((m,d));
        if roa: m = np.hstack((m,dd));
        if logfb: m = np.hstack((m,S));
        return m;
    elif method=='librosa':
        a = np.array(y,dtype=np.float32);
        D = np.abs(librosa.stft(a, window='hamming',
                    n_fft=n_fft, win_length=win_length,
                    hop_length=hop_length))**2;
        S = librosa.feature.melspectrogram(S=D, y=a, n_mels=n_mels,fmin=0,fmax=None)
        m = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)

        if roc or roa: d = librosa.feature.delta(m);
        if roa: dd = librosa.feature.delta(d);
        if roc: m = np.vstack((m,d));
        if roa: m = np.vstack((m,dd));
        if logfb: m = np.vstack((m,S));
        return m.T

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 06:06:41 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import librosa
import numpy as np
from sphfile import SPHFile
from python_speech_features import mfcc

def read_sph(path):
    assert os.path.exists(path), "SPH file not readable"
    sph = SPHFile(path);
    return sph.format['sample_rate'], sph.content,

def extract_mfcc_features(y,sr,n_mfcc=13,roc=False,roa=False):
    """
    fmax has been default to 8k allowing 4khz audio frequency samples
    """
    m = mfcc(y,samplerate=sr,numcep=n_mfcc);
    if roc or roa: d = librosa.feature.delta(m);
    if roa: dd = librosa.feature.delta(d);
    if roc: m = np.hstack((m,d));
    if roa: m = np.hstack((m,dd));
    return m

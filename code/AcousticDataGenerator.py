# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:58:07 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import math
import logging
import numpy as np
from Utils import gen_bidi_map, is_array_or_list
from AudioUtils import read_sph, extract_mfcc_features
from TrainUtils import batch_temporal_categorical

class AcousticDataGenerator:
    
    _WB_STR = '<wb>';
    _CTC_BLANK = '_';
    
    def __init__(self,corpus,mode="phoneme",output="boundary",ctc_mode=False,
                 mbatch_size=32,audio_feature='mfcc',
                 sgram_step=10,sgram_window=20,sgram_freq=8000,
                 n_mfcc=13,mfcc_win=0.025,mfcc_step=0.010,mfcc_roca=True):
        assert audio_feature=='mfcc' or audio_feature=='spectrogrm', \
        "Unsupported audio feature request "+audio_feature+\
        ". Supported are {mfcc, spectrogram}"
                
        if output!="boundary" and not ctc_mode:
            raise ValueError("Sequence output is only supproted in CTC mode");
        
        self.corpus = corpus;
        self.audio_feature = audio_feature;
        self.sgram_step = sgram_step;
        self.sgram_window = sgram_window;
        self.sgram_freq = sgram_freq;
        self.n_mfcc = n_mfcc;
        self.mfcc_roca = mfcc_roca;
        self.mode = mode;
        self.output = output;
        self.ctc_mode=ctc_mode;
        self.mbatch_size = mbatch_size;
        self.bsymbol = '<wb>';
        self.nbsymbol = '<nb>'; self.ctc_char='_';
        self.mfcc_win=mfcc_win;
        self.mfcc_step=mfcc_step;
        self.init_splits();
        self.init_output_map();
        
    def init_splits(self):
        self.train_idxs = self.corpus.get_split_ids('training');
        self.valid_idxs = self.corpus.get_split_ids('validation');
        self.test_idxs = self.corpus.get_split_ids('testing');
        self.shuffle_split('training')
        self.shuffle_split('validation')
        self.shuffle_split('testing')
        self.ibtrain = 0;
        self.ibvalid = 0;
        self.ibtest = 0;
        self.n_train = len(self.train_idxs);
        self.n_valid = len(self.valid_idxs);
        self.n_test  = len(self.test_idxs);
        
    def init_output_map(self):
        if self.output=="sequence":
            if self.mode=="phoneme":
                labels=self.corpus.get_vocab_phoneme_list();
            elif self.mode=="grapheme":
                labels=list("abcdefghijklmnopqrstuvwxyz'")
            else:
                raise ValueError("Mode should be phoneme or grapheme")
            labels.append(' ');
        elif self.output=="boundary":
            labels=[self.nbsymbol,self.bsymbol];
        else:
            raise ValueError("Output type should be {boundary, sequence}")
        if self.ctc_mode:
            labels.append(self.ctc_char);
        self.outmap = gen_bidi_map(labels);
        self.n_classes = len(self.outmap[0]);
        
    def fit_train(self,n_samples=np.Inf,eps=1e-12):
        logging.info("Computing data standardization with %d samples" % n_samples);
        data = self.get_split_data('training');
        n = min(n_samples,len(data));
        np.random.shuffle(data);
        features = [self.get_audio_features(sph)[2] for sph,seq in data[0:n]]
        stacked = np.vstack(features);
        self.feature_mean = np.mean(stacked,axis=0);
        self.feature_std = np.std(stacked,axis=0);
        self.feature_dim = len(self.feature_mean);
        #for i in range(self.feature_dim):
        #    logging.info("Feature(%d) mean: %f, std: %f",
        #             i,self.feature_mean[i],self.feature_std[i]);
        logging.info("Computed mean and std for feature standardization")
    
    def fit(self,feature,eps=1e-12):
        assert hasattr(self,'feature_mean') and hasattr(self,'feature_std'),\
        "Training data feature mean and std not initialized"
        return (feature-self.feature_mean)/(self.feature_std+eps)
    
    def get_split_idxs(self,split):
        assert split=="training" or split=="validation" or split=="testing",\
        "Unknown split's data requested. Valid {training validation testing}"
        idxs = self.train_idxs if split=="training" else \
               self.valid_idxs if split=="validation" else \
               self.test_idxs
        return idxs;
        
    def shuffle_split(self,split):
        if split=='training':
            np.random.shuffle(self.train_idxs);
        elif split=='validation':
            np.random.shuffle(self.valid_idxs);
        elif split=='testing':
            np.random.shuffle(self.test_idxs);
        else:
            raise ValueError("Split should be {training, validation, testing}")
            
    def get_split_data(self,split,idxs=[]):
        if not is_array_or_list(idxs) or len(idxs)==0:
            idxs = self.get_split_idxs(split);
        return self.corpus.get_corpus_data(idxs,self.mode);
        
    def get_audio_features(self,file):
        sr,y = read_sph(file);
        if self.audio_feature=='mfcc':
            features = extract_mfcc_features(y,sr,n_mfcc=self.n_mfcc,
                                             roca=self.mfcc_roca);
        return sr,len(y),features;
    
    def encode_output(self,seqdf,sr,input_length):
        assert hasattr(self,'outmap'), "Output map not initialized";
        if self.ctc_mode:
            opseq = self.encode_ctc_output(seqdf[2].values)
        else:
            opseq = self.encode_ce_output(sr,input_length,seqdf)
        return opseq;
        
    def encode_ctc_output(self,seq):
        assert hasattr(self,'outmap'), "Output map not initialized";
        opseq = [];
        if self.output=="boundary":
            opseq = [self.wb for i in range(0,len(seq)+1)]
        elif self.output=="sequence":
            for x in seq:
                opseq.append(self.outmap[0][x]);
                opseq.append(self.outmap[0][' ']);
            opseq.pop();
        else:
            raise ValueError("Output mode should be {boundary or sequence}")
        return opseq
               
    def encode_ce_output(self,sr,nseq,seqdf):
        """
        Nw = (L-W)/S+1
        Start(0) = 0, Start(1) = Start(0)+S=S, Start(2)=Start(1)+S=S+S=2S
        => Start(w) = w*S;
        Given an index i in original signal, which windows does it fall in
        Nw(i)=(i-W)/S+1 --> Taking time resorting to for loop 
        Ignore h# as it denotes silence in phoneme mode
        Only left boundary detection coded -- There are silence gaps
        would warrant right boundary detection as well -- TBD
        """
        assert hasattr(self,'outmap'), "Output map not initialized";
        ns_win = math.ceil(self.mfcc_win*sr);
        ns_step = math.ceil(self.mfcc_step*sr);
        sphone = self.corpus.get_silence_phoneme();
        if self.mode=="phoneme":
            idxs = [0,seqdf.index[-1]]
            for i in idxs:
                if seqdf.loc[i][2]==sphone: seqdf.drop(index=i);
        bnds=seqdf[0].values; opseq=[];
        for i in range(nseq):
            start = i*ns_step; end = start+ns_win;
            has_bnd = np.dot(bnds>start,bnds<end);
            symbol = self.bsymbol if has_bnd else self.nbsymbol
            opseq.append(self.outmap[0][symbol]);
        return opseq;
    
    def gen_split_batch(self,split,idxs):
        data=self.get_split_data(split,idxs);
        bfeats=[]; iplen=[]; oplen=[]; labels=[];
        for a,seqdf in data:
            sr,n_as,feats=self.get_audio_features(a);
            bfeats.append(self.fit(feats));
            iplen.append(len(bfeats[-1]));
            opseq = self.encode_output(seqdf,sr,iplen[-1]);
            oplen.append(len(opseq));
            labels.append(opseq);
        max_iplen=max(iplen); max_oplen=max(oplen);
        pad_label=0 if self.output=="boundary" else \
                    list(self.outmap[0].values())[-1];
        X = np.zeros([self.mbatch_size,max_iplen,bfeats[-1].shape[1]]);
        Y = np.ones([self.mbatch_size,max_oplen])*pad_label;
        for i in range(0,self.mbatch_size):
            feats=bfeats[i]; X[i,0:feats.shape[0],:]=feats;
            Y[i,0:len(labels[i])]=labels[i];
        if self.ctc_mode:
            outputs = {'ctc': np.zeros(self.mbatch_size)}
            inputs = {'the_input': X, 
                      'the_labels': Y, 
                      'input_length': iplen, 
                      'label_length': oplen 
                      }
        else:
            outputs=batch_temporal_categorical(Y,self.n_classes);
            inputs=X;
        return (inputs, outputs)
            
    def train_generator(self):
        while True:
            idxs = self.train_idxs[self.ibtrain:self.mbatch_size]
            mbatch = self.gen_split_batch('training',idxs)
            self.ibtrain += self.mbatch_size;
            if self.ibtrain >= self.n_train:
                self.ibtrain = 0; self.shuffle_split('training')
            yield mbatch

    def valid_generator(self):
        while True:
            idxs = self.valid_idxs[self.ibvalid:self.mbatch_size]
            mbatch = self.gen_split_batch('validation',idxs)
            self.ibvalid += self.mbatch_size;
            if self.ibvalid >= self.n_valid:
                self.ibvalid = 0; self.shuffle_split('validation')
            yield mbatch
        
    def test_generator(self):
        while True:
            idxs = self.test_idxs[self.ibtest:self.mbatch_size]
            mbatch = self.gen_split_batch('testing',idxs)
            self.ibtest += self.mbatch_size;
            if self.ibtest >= self.n_test:
                self.ibtest = 0; self.shuffle_split('testing')
            yield mbatch
    
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 09:52:43 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from Utils import *
from glob import glob1
from sphfile import SPHFile
import matplotlib.pyplot as plt

class TimitDataset:
    
    _COMMENT_RE = re.compile("^;");
    _SPKR_COLS = ['id','sex','dr','use','recdate','birthdate',
                  'ht','race','edu','comments'];
    _DR_RE = re.compile('^dr\d+$');
    _DR_PREFIX = 'dr';
    
    def __init__(self,root):
        assert os.path.exists(root), "Root folder does not exist"
        self._root = root;
        vocab = os.path.join(root,"doc","TIMITDIC.TXT")
        spkrinfo = os.path.join(root,"doc","SPKRINFO.TXT")
        prompts = os.path.join(root,"doc","PROMPTS.TXT")
        self.init_dictionary(vocab);
        self.init_spkrinfo(spkrinfo);
        self.init_sentences(prompts);
        self.init_files();
        
    def _is_comment(self,line):
        return self._COMMENT_RE.search(line)!=None
    
    def init_dictionary(self,vocab):
        logging.info('Start parsing dictionary')
        assert os.path.exists(vocab), "Missing vocab dict: "+vocab
        f = open(vocab, 'r');
        linecnt = 0; rows = [];
        for line in list(f):
            linecnt+=1;
            if self._is_comment(line): continue
            rline=re.sub("/","",line);
            rline=re.sub("\d+","",rline);
            wlist = rline.split();
            if len(wlist)<2:
                msg = 'Incomplete dict entry @%d : %s'
                logging.warn(msg,linecnt,line); continue
            rows.append([wlist[0], ' '.join(wlist[1:])])
        f.close();
        df = pd.DataFrame(data=rows,columns=["word","phnseq"]);
        assert df.shape[0]>0, "Invalid dictionary no valid entry found"
        self._vocab = vocab; self._vocabdf = df;
        df.set_index('word',inplace=True);
        logging.info("Read %d words from dictionary",df.shape[0])
        
    def init_spkrinfo(self,spkrinfo):
        logging.info('Start parsing speaker information')
        assert os.path.exists(spkrinfo), "Missing speaker info: "+spkrinfo
        f = open(spkrinfo,"r"); linecnt=0; rows=[];
        for line in list(f):
            linecnt+=1;
            if self._is_comment(line): continue
            wlist = line.split();
            if len(wlist)<9:
                msg = 'Incomplete speaker entry @%d : %s'
                logging.warn(msg,linecnt,line); continue
            row = wlist[0:9]; row.append(' '.join(wlist[9:]));
            row[0]=row[0].lower();
            rows.append(row);
        f.close()
        assert len(rows)>0, "No valid speaker entry found"
        df = pd.DataFrame(data=rows,columns=self._SPKR_COLS);
        df.set_index('id',inplace=True);
        self._spkrinfo = spkrinfo; self._spkrdf = df;
        
    def init_sentences(self,prompts):
        assert os.path.exists(prompts), "Missing sentence files: "+prompts
        f = open(prompts,"r"); linecnt=0; rows=[];
        for line in list(f):
            linecnt+=1;
            if self._is_comment(line): continue
            r = re.compile('\(.+\)');
            if not r.search(line):
                msg = 'sentence id not found @%d %s';
                logging.warn(msg,linecnt,line);
                continue;
            wlist = line.split();
            i = re.sub('[()]',"",wlist[-1]);
            c = re.sub('[()\d]',"",wlist[-1]);
            row = [i,c,' '.join(wlist[0:-1])];
            rows.append(row);
        f.close();
        assert len(rows)>0, "No valid sentence found"
        logging.info('Read %d sentences',len(rows));
        df = pd.DataFrame(data=rows,columns=['id','type','sentence']);
        df.set_index('id',inplace=True);
        self._sentfile = prompts; self._sentdf = df;
        
    def get_dialect_regions(self):
        assert hasattr(self,'_spkrdf'), "Speaker info is not initialized"
        return ['dr'+x for x in self._spkrdf.dr.unique()];
    
    def has_speaker(self,spkr):
        assert hasattr(self,'_spkrdf'), "Speaker info is not initialized"
        return spkr in self._spkrdf.index
    
    def get_region_id(self,name):
        if self._DR_RE.search(name): return name[2:];
    
    def init_files(self):
        dirs = glob1(self._root,'dr*'); dirs+=glob1(self._root,'DR*');
        rows = [];
        assert len(dirs)>0, "No dialect region directory division found dr*"
        for drd in dirs:
            drid = int(self.get_region_id(drd));
            drp = os.path.join(self._root,drd);
            # First character is 'f' - female, 'm'- male
            spkrdirs = glob1(drp,'[fmFM]*');
            for spkd in spkrdirs:
                sex = spkd[0]; spkr = spkd[1:];
                spkp = os.path.join(drp,spkd);
                # Get waves and check for wrd and phn files
                wavfiles = glob1(spkp,'*.wav');
                for wav in wavfiles:
                    senid = wav[0:-4];
                    phn = senid+'.phn'; wrd = senid+'.wrd';
                    phnp = os.path.join(spkp,phn);
                    wrdp = os.path.join(spkp,wrd);
                    if not (os.path.exists(phnp) and os.path.exists(wrdp)):
                        logging.warn('Could not find wrd or phn file '+spkp);
                        continue;
                    row = [drid,spkr,sex,senid,wav,phn,wrd];
                    rows.append(row);
        assert len(rows)>0, "No valid data found in dataset "+self._root;
        cols = ['dr','spkrid','sex','senid','wav','phn','wrd'];
        df = pd.DataFrame(rows,columns=cols);
        logging.info('Total %d valid samples found in dataset',df.shape[0])
        self._corpusdf = df;
        return
    
    def get_grapheme_list(self):
        """
        Parse sentence information to collect unique chars
        Exclude list of 
        in Acoustic model and can be taken care of through Language model
        """
        assert hasattr(self,'_sentdf'), "Sentence info is not initialized"
        cdups = [];
        for x in self._sentdf.sentence: cdups+=list(set(x));
        cdups = [x.lower() for x in cdups]
        return list(set(cdups))
    
    def get_phoneme_list(self):
        assert hasattr(self,'_vocabdf'), "Vocabulary is not initialized"
        phns = [];
        for x in self._vocabdf.phnseq: phns+=x.split();
        return list(set(phns));
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
from glob import glob1
from PlotUtils import *

class Timit:
    
    _COMMENT_RE = re.compile("^;");
    _SPKR_COLS = ['id','sex','dr','use','recdate','birthdate',
                  'ht','race','edu','comments'];
    _DR_RE = re.compile('^dr\d+$');
    _DR_PREFIX = 'dr';
    
    def __init__(self,root,verbose=False):
        assert os.path.exists(root), "Root folder does not exist"
        logging.info('Initializing Timit corpus from '+root);
        self._root = root;
        vocab = os.path.join(root,"doc","TIMITDIC.TXT")
        spkrinfo = os.path.join(root,"doc","SPKRINFO.TXT")
        prompts = os.path.join(root,"doc","PROMPTS.TXT")
        self.init_dictionary(vocab);
        self.init_spkrinfo(spkrinfo);
        self.init_sentences(prompts);
        self.init_files(verbose=verbose);
        self.silence_phoneme = 'h#';
        
    def get_silence_phoneme(self):
        return self.silence_phoneme
    
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
        logging.info('Read information for %d speakers',df.shape[0]);
        
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
    
    def get_speaker_use(self,spkr):
        if self.has_speaker(spkr):
            return self._spkrdf.loc[spkr]['use']
    
    def get_region_id(self,name):
        if self._DR_RE.search(name): return name[2:];
    
    def init_files(self,verbose=False):
        dirs = glob1(self._root,'dr*');
        # May need this for linux but windows is case insensitive
        #dirs+=glob1(self._root,'DR*');
        rows = []; f = open('timit_corpus_parsing.log',mode='w');
        assert len(dirs)>0, "No dialect region directory division found dr*"
        logging.info("Start initializing corpus files")
        for drd in dirs:
            logging.info("Parsing files for dialect dir %s",drd);
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
                    senid = wav[0:-4]; wavp = os.path.join(spkp,wav);
                    phn = senid+'.phn'; wrd = senid+'.wrd';
                    phnp = os.path.join(spkp,phn);
                    wrdp = os.path.join(spkp,wrd);
                    if not (os.path.exists(phnp) and os.path.exists(wrdp)):
                        logging.warn('Could not find wrd or phn file '+spkp);
                        continue;
                    row = [drid,spkr,sex,senid,wavp,phnp,wrdp]
                    # Check for overlap in wrd and phn both and report
                    if self.has_overlap(phnp):
                        msg = "Phone boundaries overlap. "+ \
                                     "Dropping entry %s" % str(row)
                        if verbose: logging.warn(msg);
                        f.write(msg+"\n");
                    elif self.has_overlap(wrdp):
                        msg = "Word boundaries overlap. "+\
                                     "Dropping entry %s" % str(row)
                        if verbose: logging.warn(msg);
                        f.write(msg+"\n");
                    else:
                        spkr_use = self.get_speaker_use(spkr);
                        train = True if spkr_use=='TRN' else False
                        test = not train; valid = False;
                        row+=[train,valid,test];
                    rows.append(row);
        assert len(rows)>0, "No valid data found in dataset "+self._root;
        cols = ['dr','spkrid','sex','senid','wav','phn','wrd',
                'training','validation','testing'];
        df = pd.DataFrame(rows,columns=cols);
        count = np.count_nonzero(df[['training','testing']].values)
        logging.info('Total %d valid samples found in dataset',count)
        self._corpusdf = df; f.close();
        return
    
    def has_overlap(self,path):
        assert os.path.exists(path), "Cannot open file " + path
        df = pd.read_csv(path,header=None,delim_whitespace=True);
        a = np.ndarray.flatten(df[[0,1]].values);
        return not np.all(np.diff(a)>=0);
        
    def get_grapheme_list(self):
        """
        Parse sentence information to collect unique chars
        """
        self.check_sentences_init();
        cdups = [];
        for x in self._sentdf.sentence: cdups+=list(set(x));
        cdups = [x.lower() for x in cdups]
        return list(set(cdups))
    
    def get_vocab_phoneme_list(self):
        assert hasattr(self,'_vocabdf'), "Vocabulary is not initialized"
        phns = [];
        for x in self._vocabdf.phnseq: phns+=x.split();
        return list(set(phns));
    
    def get_phoneme_list(self):
        assert hasattr(self,'_corpusdf'), "Corpus is not initialized"
        phns = [];
        for f in self._corpusdf.phn.values:
            df = pd.read_csv(f,header=None,delim_whitespace=True);
            phns += list(df[2].values)
        return list(set(phns))
    
    def report_statistics(self,prefix='timit',folder=None,
                          reptext=True,plotfig=True):
        self.check_corpus_init();
        combos = [['Total','training==True or training==False or training!=training'],
                  ['Dropped','training!=training and testing!=testing and validation!=validation'],
                  ['Training','training==True'],
                  ['Validation','validation==True']]
        for name,expr in combos:
            df = self._corpusdf.query(expr);
            drsum = df.groupby(by='dr').size();
            if reptext:
                print(name+" corpus statistics")
                print("==========================")
                print(drsum);
            if plotfig and drsum.size>0:
                fig = new_figure();
                drsum.plot.pie(title='Dialect region wise '+
                               name.lower()+' count',
                               label='Count',table=True);
                save_figure(prefix+'.'+name.lower(),folder=folder);
            
    
    def check_corpus_init(self):
        assert hasattr(self,'_corpusdf'), "Corpus is not initialized"
        
    def check_sentences_init(self):
        assert hasattr(self,'_sentdf'), "Sentence info is not initialized"
        
        
    def split_validation(self,train_ratio=0.9):
        self.check_corpus_init();
        drs = np.unique(self._corpusdf.dr.values);
        for dr in drs:
            for sex in ['m','f']:
                expr = 'dr==@dr and sex==@sex and training==True';
                df = self._corpusdf.query(expr);
                total = df.shape[0]; valcnt = int((1-train_ratio)*total);
                msg='Creating %d validation samples for gender=%s and dr=%s'
                logging.info(msg,valcnt,sex,dr);
                if valcnt>0:
                    idxs = df.sample(n=valcnt).index;
                    self._corpusdf.loc[idxs,'validation']=True;
                    self._corpusdf.loc[idxs,'training']=False;
                    
    def strip_punctuations(self,s):
        # Retain hypens and dots after
        schomp = re.sub('\.$','',s);
        return re.sub('[?!:;,"]','',schomp)
    
    def get_words(self,sentences):
        words = [];
        for x in sentences:
            words += self.strip_punctuations(x).split();
        return [x.lower() for x in list(set(words))];
    
    def get_all_sentences(self):
        self.check_sentences_init();
        return self._sentdf.sentence.values;
    
    def get_sentences(self,idxs):
        self.check_sentences_init();
        return self._sentdf.loc[idxs,'sentence'].values;
    
    def get_train_sentence_ids(self):
        self.check_corpus_init();
        idxs = self._corpusdf.query('training==True')['senid'];
        idxs = list(set(idxs));
        return idxs;

    def get_train_sentence_count(self):
        return len(self.get_train_sentence_ids());
    
    def get_train_sentences(self):
        self.check_sentences_init();
        idxs = self.get_train_sentence_ids();
        return self.get_sentences(idxs);
    
    def get_all_words(self):
        self.check_sentences_init();
        return self.get_words(self.get_all_sentences());
        
    def get_train_words(self):
        self.check_sentences_init();
        sents = self.get_train_sentences();
        return self.get_words(sents);
        
    def get_sentence_count(self):
        self.check_sentences_init();
        return self._sentdf.shape[0]
    
    def report_train_coverage(self,report='train_coverage.rpt'):
        sents_cnt = self.get_sentence_count();
        trsents_cnt = self.get_train_sentence_count();
        words = self.get_all_words();
        trwords = self.get_train_words();
        nw = len(words); ntw = len(trwords);
        ptw = round(ntw/nw*100,2);
        pts = round(trsents_cnt/sents_cnt*100,2);
        logging.info("Analyzing training set coverage");
        fh = open(report, "w");
        fh.write("Training coverage report\n");
        fh.write("========================\n");
        fh.write("Total sentences = {}\n".format(sents_cnt));
        fh.write("Train sentences = {} ({})\n".format(trsents_cnt,pts));
        fh.write("Total words     = {}\n".format(nw));
        fh.write("Train words     = {} ({})\n".format(ntw,ptw));
        missing = [];
        for x in words:
            if x not in trwords: missing.append(x);
        fh.write("List of words missing in training set\n")
        fh.write("-"*80+"\n");
        fh.write("\n".join(missing))
        fh.close();
        logging.info("Written training set coverage report to %s",report);
        
    def get_split_ids(self,split):
        self.check_corpus_init();
        assert split=='training' or split=='testing' or split=='validation',\
        "Incorrect split requested - choose {training testing validation}"
        return self._corpusdf.query(split+'==True').index.values;
    
    def get_corpus_columns(self,idxs,keys):
        self.check_corpus_init();
        return self._corpusdf.loc[idxs,keys]
    
    def get_corpus_data(self,idxs):
        keys = ['wav','phn','wrd'];
        df = self.get_corpus_columns(idxs,keys).values;
        flist=[];
        for wav,pseqf,wseqf in df:
            pdf = pd.read_csv(pseqf,header=None,delim_whitespace=True);
            wdf = pd.read_csv(wseqf,header=None,delim_whitespace=True)
            #seq = np.ndarray.flatten(df.values);
            flist.append([wav,pdf,wdf]);
        return flist;
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 15:19:44 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import sys
import logging
import numpy as np
from glob import glob
from time import gmtime, strftime

def create_folder(folder):
    if not os.path.exists(folder): os.makedirs(folder,exist_ok=True)

def curtime():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def initlog(logfile,stdout=True,fmt=None,level=logging.INFO):
    if not fmt:
        fmt = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    logfmt = logging.Formatter(fmt);
    rootlog = logging.getLogger();
    fileHandler = logging.FileHandler(logfile);
    fileHandler.setFormatter(logfmt)
    rootlog.addHandler(fileHandler)
    if stdout:
        conshandler = logging.StreamHandler(sys.stdout)
        conshandler.setFormatter(logfmt)
        rootlog.addHandler(conshandler)
    rootlog.level=level;
    return rootlog

def print_bold(text):
    print('\x1b[1;31m'+text+'\x1b[0m')
    
def get_file_counter(folder,glob_mask='*'):
    counter = 0;
    if folder!=None and os.path.exists(folder):
        files = glob(folder+'/'+glob_mask);
        files = [os.path.split(f)[1] for f in files]
        counter = 0 if (len(files)==0) else \
        max([int(os.path.splitext(f)[0]) for f in files])
    return counter

def gen_bidi_map(elist):
    bmap = [{}, {}]; n = len(elist);
    bmap[0] = {elist[i]:i for i in range(n)}
    bmap[1] = {i:elist[i] for i in range(n)}
    return bmap;

def is_array_or_list(v):
    return isinstance(v,(np.ndarray,list))

def dump_data(data,filepath):
    import _pickle as pickle
    f = open(filepath,'wb');
    pickle.dump(data,f);
    f.close();


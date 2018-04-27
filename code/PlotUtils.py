# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 07:15:09 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt

from Utils import *

def init_sns_style(style='white'):
    sns.set_style('white')
    
def new_figure(size=None):
    return plt.figure();

def save_figure(prefix,folder=None,size=None,imgfmt='jpg'):
    assert size==None or len(size)==1 or len(size)==2, "Size should be tuple"
    if size:
        tup = size if len(size)==2 else (size,size);
        plt.gcf().set_size_inches(tup[0], tup[1], forward=True)
    fn = '.'.join([prefix,imgfmt]);
    if folder:
        create_folder(folder);
        fn = os.path.join(folder,fn);
    plt.savefig(fn);
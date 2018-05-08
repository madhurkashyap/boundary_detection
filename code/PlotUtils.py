# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 07:15:09 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import math
import numpy as np
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
    
def plot_keras_history(history,keydict,suptitle='',boxsize=4,
                       legendloc='lower center'):
    '''
    Accepts history dictionary object as input. Plots both training
    and validation accuracy and loss curves against epochs
    '''

    for keytup in keydict.values():
        for key in keytup:
            if not key in history: raise KeyError(key);

    x = list(range(len(history[key])));
    
    nfigs = len(keydict);
    ncols= 2 if nfigs>=2 else 1;
    nrows = math.ceil(nfigs/2);
    figsize = (nrows*boxsize,ncols*boxsize);
    f, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize);
    f.suptitle(suptitle);
    if not isinstance(axes,np.ndarray): axes = [axes];
    axes = np.ndarray.flatten(axes);
    
    titles = list(keydict.keys());
    for i in range(nfigs):
        ax = axes[i]; ax.set_xlabel('# Epoch');
        ax.set_ylabel(titles[i]);
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(9)
        for key in keydict[titles[i]]:
            line, = ax.plot(x,history[key],label=key);
        ax.legend();

    plt.tight_layout(h_pad=0.9)
    plt.show()

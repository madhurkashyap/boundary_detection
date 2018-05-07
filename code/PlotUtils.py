# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 07:15:09 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
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
    
def plot_keras_history(history,suptitle='',figsize=(8,4),
                       legendloc='lower center'):
    '''
    Accepts history dictionary object as input. Plots both training
    and validation accuracy and loss curves against epochs
    '''

    if not 'acc' in history and not 'loss' in history:
        raise ValueError("Neither loss nor accuracy data");

    x = list(range(len(history['loss'])));

    data = {}; titles = []; figdata = [];
    if 'loss' in history: figdata.append(['loss','Train','b']);
    if 'val_loss' in history: figdata.append(['val_loss','Test','r']);
    if len(figdata)>0: data['Loss']=figdata;
    
    figdata = [];
    if 'acc' in history: figdata.append(['acc','Train','b'])
    if 'val_acc' in history: figdata.append(['val_acc','Test','r']);
    if len(figdata)>0: data['Accuracy']=figdata;

    if len(data)==0: return;
    titles = list(data.keys())
    f, axes = plt.subplots(nrows=1,ncols=len(data),figsize=figsize);
    f.suptitle(suptitle);
    if not isinstance(axes,np.ndarray): axes = [axes];
    
    for i in range(len(data)):
        ax = axes[i]; legpts = []; legtxts = [];
        ax.set_xlabel('# Epoch');
        ax.set_ylabel(titles[i]);
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(9)
        for key,legtxt,color in data[titles[i]]:
            line, = ax.plot(x,history[key],color,label=legtxt);
            legpts.append(line); legtxts.append(legtxt);
    
    f.legend(legpts,legtxts,legendloc,
             fontsize='small',ncol=2,frameon=False)

    plt.tight_layout(h_pad=0.9)
    plt.show()

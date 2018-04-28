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
    
def plot_keras_history(history,suptitle=''):
    '''
    Accepts history dictionary object as input. Plots both training
    and validation accuracy and loss curves against epochs
    '''
    x = list(range(len(history['acc'])));
    f, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(7,3))
    f.suptitle(suptitle)
    #ax1.set_title('Loss')
    ax1.set_xlabel('# Epoch')
    ax1.set_ylabel('Loss')
    #ax2.set_title('Accuracy')
    ax2.set_xlabel('# Epoch')
    ax2.set_ylabel('Accuracy')

    y1a=history['loss'];
    y1b=history['acc'];
    line1, = ax1.plot(x,y1a,'b',label="Train")
    line2, = ax2.plot(x,y1b,'b',label="Train")
    if 'val_acc' in history.keys():
        y2a=history['val_loss'];
        y2b=history['val_acc'];
        line3, = ax1.plot(x,y2a,'r',label="Test")
        line4, = ax2.plot(x,y2b,'r',label="Test")
        legpt = (line1,line3); legtxt = ('Train','Test');
    else:
        legpt = [line1]; legtxt = ['Train'];
    f.legend(legpt,legtxt,'upper center',
                 fontsize='small',ncol=2,frameon=False)
    for ax in [ax1,ax2]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(9)
    plt.tight_layout(h_pad=0.9)
    plt.show()
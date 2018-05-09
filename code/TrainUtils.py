# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 08:14:12 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import numpy as np
import pandas as pd
from copy import copy
import keras.backend as K
from Utils import create_folder, dump_data
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input, Lambda
from sklearn.metrics import confusion_matrix

def weighted_categorical_crossentropy(target,output,weights=1.0):
    # scale preds so that the class probas of each sample sum to 1
    output /= K.tf.reduce_sum(output,
                                len(output.get_shape()) - 1,
                                True)
    
    class_weights = K.tf.constant(weights);
    _epsilon = K.tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
    output = K.tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return  -K.tf.reduce_sum(K.tf.multiply(target * K.tf.log(output),class_weights),
                           len(output.get_shape()) - 1)

def ctc_func(args):
    y_pred, y_true, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def add_ctc(model):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(model.output_length)(input_lengths)
    ctc_loss = Lambda(ctc_func, output_shape=(1,), name='ctc')(
        [model.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[model.input, the_labels, input_lengths, label_lengths], 
        outputs=ctc_loss)
    return model

def batch_temporal_categorical(y,n_classes):
    assert len(y.shape)==2, "Temporal batch predictions should be 2-dimensional"
    yoh = np.zeros((y.shape[0],y.shape[1],n_classes))
    for i in range(y.shape[0]):
        oh = [to_categorical(x,n_classes) for x in y[i]]
        yoh[i]=oh;
    return yoh;

def train_model(model,trgen,valgen,prefix,
                epochs=1,verbose=1,loss='binary_crossentropy',
                history_folder='./history',model_folder='./models',
                optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, 
                    nesterov=True, clipnorm=5),
                steps_per_epoch=100, validation_steps=10,
                metrics = ['acc'],save_period=0,
                sample_weight_mode=None,
                report_stats=False,
                class_names=[]):

    model_path = os.path.join(model_folder,prefix+'.{epoch:02d}-{val_loss:.2f}.hdf5')
    pickle_path = os.path.join(history_folder,prefix+'.pkl');
    create_folder(model_folder); create_folder(history_folder);
    callbacks = [];
    if save_period>0:
        callbacks.append(ModelCheckpoint(filepath=model_path,
                                         period=save_period,verbose=1));
    if loss=='ctc':
        ctcmodel = add_ctc(model);
        ctcmodel.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, 
                optimizer=optimizer, sample_weight_mode=sample_weight_mode);
        hist = ctcmodel.fit_generator(generator=trgen,
                               validation_data=valgen,
                               epochs=epochs,
                               steps_per_epoch=steps_per_epoch,
                               validation_steps=validation_steps,                           
                               callbacks=callbacks,
                               verbose=verbose,shuffle=False);
    else:    
        model.compile(optimizer=optimizer,loss=loss,metrics=metrics,
                sample_weight_mode=sample_weight_mode);
        hist = model.fit_generator(generator=trgen,
                               validation_data=valgen,
                               epochs=epochs,
                               steps_per_epoch=steps_per_epoch,
                               validation_steps=validation_steps,                           
                               callbacks=callbacks,
                               verbose=verbose,shuffle=False);
        if report_stats:
            cnf = get_model_class_stats(model,valgen,validation_steps,class_names);
    dump_data(hist.history,pickle_path);
    return hist

def precision(yt,yp,idx,n_classes,thresh=0):
    # Not implemented how to use this threshold
    ypcls = K.reshape(K.argmax(yp,axis=2),[-1]);
    ytcls = K.reshape(K.argmax(yt,axis=2),[-1]);
    labels = list(range(n_classes));
    cnf = K.tf.confusion_matrix(ytcls,ypcls)
    rest = copy(labels); rest.remove(idx);
    tp = cnf[idx,idx]; fp = 0;
    for r in rest: fp += cnf[r,idx];
    return tp/(tp+fp);

def recall(yt,yp,idx,n_classes,thresh=0):
    # Not implemented how to use this threshold
    ypcls = K.reshape(K.argmax(yp,axis=2),[-1]);
    ytcls = K.reshape(K.argmax(yt,axis=2),[-1]);
    labels = list(range(n_classes));
    cnf = K.tf.confusion_matrix(ytcls,ypcls);
    rest = copy(labels); rest.remove(idx);
    tp = cnf[idx,idx]; fn = 0;
    for r in rest: fn += cnf[idx,r];
    return tp/(tp+fn);

def clsacc(yt,yp,idx,n_classes,thresh=0):
    # Not implemented how to use this threshold
    ypcls = K.reshape(K.argmax(yp,axis=2),[-1]);
    ytcls = K.reshape(K.argmax(yt,axis=2),[-1]);
    labels = list(range(n_classes));
    cnf = K.tf.confusion_matrix(ytcls,ypcls);
    rest = copy(labels); rest.remove(idx);
    tp = cnf[idx,idx]; total = K.tf.reduce_sum(cnf[idx,:])
    return tp/(total);
    
def get_model_class_stats(model,generator,steps,names=[]):
    X,y = next(generator)
    n_classes=y.shape[-1];
    cnf = np.zeros((n_classes,n_classes));
    labels = list(range(n_classes));
    if (len(names)>0 and len(names)!=n_classes):
        raise ValueError("Please specify names of all classes or none");
    if len(names)==0: names = [str(i) for i in range(n_classes)]
    for j in range(steps):
        X,y = next(generator)
        yp = model.predict(X);
        ypcls = np.argmax(yp,axis=2);
        ytcls = np.argmax(y,axis=2);    
        for i in range(len(ypcls)):
            cnf += confusion_matrix(ytcls[i], ypcls[i], labels=labels);

    rows = [];
    for i in range(n_classes):
        rest = copy(labels); rest.remove(i);
        row = [names[i],np.sum(cnf[i,:]),cnf[i,i],np.sum(cnf[rest,i]),
               np.sum(cnf[i,rest])]
        #print("Total %s: %d" % (name,np.sum(cnf[i,:])));
        #print("Total true positive %s: %d" % (name,cnf[i,i]));
        #print("Total false positive %s: %d" % (name,np.sum(cnf[rest,i])));
        #print("Total false negative %s: %d" % (name,np.sum(cnf[i,rest])));
        rows.append(row);
    df = pd.DataFrame(data=rows,columns=['class','count','true +ve',
                                         'false +ve','false -ve'])
    print('');
    print(df);
    return cnf;

def ctc_report_metrics(model,generator,steps,symbolmap,wlen,wstep,
                          ctc_blank='_',iplkey='input_length',
                          labkey='the_labels',bndkey='boundaries',
                          mode='conservative'):
    truep = 0; falsen = 0; falsep = 0; wcount = 0;
    tprate = 0; fnrate = 0; fprate = 0;
    for i in range(steps):
        print("\rProcessing batch %03d / %03d" % (i+1,steps),end='');
        X,yt = next(generator); count = X[iplkey].shape[0];
        yp = model.predict(X);
        for j in range(count):
            iplen = X[iplkey][j];
            ypsc = np.argmax(yp[j,0:iplen],axis=1);
            ypsl = [symbolmap[x] for x in ypsc];
            ypbnd = ctc_decode_boundaries(ypsl,wlen,wstep);
            ytbnd = X[bndkey][j].values[:,[0,1]]
            wcount += ytbnd.shape[0];
            for st,en in ytbnd:
                m = (ypbnd>=st)*1 * (ypbnd<=en)*1;
                nz = np.count_nonzero(m);
                if nz==0: falsen+=1;
                if nz==1: truep+=1;
                if nz>1: falsep+=1;
    tprate = truep/wcount; fnrate = falsen/wcount; fprate=falsep/wcount;
    print("\nTotal word windows: %d" % wcount);
    print("Correct detections: %d (%f)" % (truep,tprate))
    print("False detections: %d (%f)" % (falsep,fprate))
    print("Missed detections: %d (%f)" % (falsen,fnrate))

def ctc_decode_boundaries(yp,wlen,wstep,mode='balanced',
                          ctc_blank='_',include_end=False):
    ypa = np.array(yp);
    idx = np.reshape(np.argwhere(ypa!='_'),(-1))[0];
    bnds = [[idx]] if idx==0 else [[idx-1]]
    gi = np.reshape(np.argwhere(ypa==' '),(-1,))
    if len(gi)>0: bnds += np.split(gi, np.where(np.diff(gi) != 1)[0]+1)
    return np.array([((x[-1]+x[0])*wstep+wlen)/2 for x in bnds])
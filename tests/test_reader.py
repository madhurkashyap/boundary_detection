# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 11:32:54 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import sys
import logging

prog = os.path.basename(__file__)
codedir = os.path.join(os.path.dirname(__file__),"..","code")
sys.path.append(codedir)

from Utils import *
from PlotUtils import *
from SpeechCorpus import Timit

logfile = prog+'.log'
rootlog = initlog(logfile,level=logging.DEBUG);

rootlog.info('Starting new session');

ds = Timit(root=sys.argv[1])

init_sns_style();

rootlog.info("Testing get_region_id");
rootlog.info('dr10a - %s',ds.get_region_id("dr10a"));
rootlog.info('dr10 - %s',ds.get_region_id("dr10"));
rootlog.info('');
rootlog.info("Testing has_speaker");
rootlog.info('fwew0 - %s',ds.has_speaker('fwew0'))
rootlog.info('wew0 - %s',ds.has_speaker('wew0'))
rootlog.info('');
rootlog.info('Testing get_grapheme_list');
rootlog.info(ds.get_grapheme_list());
rootlog.info('');
rootlog.info('Testing get_phoneme_list');
rootlog.info(ds.get_phoneme_list());
rootlog.info('');
ds.report_train_coverage(report='train_coverage_before_split.rpt');
ds.split_validation();
rootlog.info('Testing report_statistics');
rootlog.info(ds.report_statistics(folder='report/images'));
ds.report_train_coverage(report='train_coverage_after_split.rpt');

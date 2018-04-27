# Madhur Kashyap 2016EEZ8350 (madhurDOTkashyap  gmail)
# Google Colaboratory Utilities for running projects on colab
# Updated: Apr 27 2018


import os
import sys
import tensorflow as tf
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

def authenticate():
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    # An interactive procedure -- asks to go to a webform and get
    # Session specific authentication
    drive = GoogleDrive(gauth)

def git_clone(user,project,base_url='https://github.com',codedirs=[]):
    url = '/'.join([base_url,user,project])
    !rm -rf {project}
    !git clone {url}

    for cdir in codedirs:
        rdir = os.path.join(".",project,cdir)
        if rdir not in sys.path:
            sys.path.append(rdir);

def verify_gpu():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU device not found')
        has_gpu = True
    else:
        print('Found GPU at: {}'.format(device_name))
        has_gpu = False
    return has_gpu

def report_resources():
    !echo '++++++++++++++++++++++++++++++++++'
    !df -h .
    !echo '++++++++++++++++++++++++++++++++++'
    !cat /proc/meminfo | grep -i mem
    !echo '++++++++++++++++++++++++++++++++++'
    !cat /proc/cpuinfo | grep -i cpu
    !echo '++++++++++++++++++++++++++++++++++'

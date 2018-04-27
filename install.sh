#!/bin/bash
#------------------------------------------------------------------------------
# Date : Apr 27 2018
# Madhur Kashyap 2016EEZ8350
# Necessary packages for running code on google colaboratory
#------------------------------------------------------------------------------


declare -a pkgs=("PyDrive" "librosa" "sphfile" "python_speech_features")

for pkg in "${pkgs[@]}"
do
    echo "Installing $pkg ..."
    pip install -U -q $pkg &> /dev/null
    if [ $? != 0 ]; then
        echo "$pkg installation failed ..."
    fi
done

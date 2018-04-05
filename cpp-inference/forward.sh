#!/bin/sh

export LD_LIBRARY_PATH=/home/cypox/phd/polsl_uvhc/caffe-net/caffe/build/lib/:$LD_LIBRARY_PATH

./forward $1 $2

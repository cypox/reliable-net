#!/usr/bin/python

import os
os.environ["GLOG_minloglevel"] = "3"

import numpy as np
import caffe



caffe.set_mode_cpu()

net = caffe.Net('network.prototxt', 'network_initialized.caffemodel', caffe.TEST)

input_shape = net.blobs['input0'].data.shape
#input_layer = np.random.rand(*input_shape)
input_layer = np.zeros(input_shape)

net.blobs['input0'].data[:] = input_layer

net.forward()

output_shape = net.blobs['output'].data.shape
output_layer = np.zeros(output_shape)
output_layer = net.blobs['output'].data

print output_layer

#!/usr/bin/python

import os
os.environ["GLOG_minloglevel"] = "3"

import numpy as np
import caffe



caffe.set_mode_cpu()

net = caffe.Net('network.prototxt', caffe.TEST)


print 'Layer outputs:'

for bl in net.blobs:
  print net.blobs[bl].data.shape

print 'Initializing weights randomely:'

for par in net.params:
  layer_shape = net.params[par][0].data.shape
  bias_shape = net.params[par][1].data.shape
  net.params[par][0].data[:] = np.random.uniform(-1, 1, layer_shape)
  net.params[par][1].data[:] = np.random.uniform(-1, 1, bias_shape)
  print layer_shape, bias_shape , ' initialized randomely.'

net.save('network_initialized.caffemodel')

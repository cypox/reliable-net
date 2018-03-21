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

print 'output before bit-flip: ', output_layer

print 'RUNTIME_ERRORS:'
print "simulating errors in the output of layer 'dense_1':"

net.forward(start='input0', end='dense_1')

net.blobs['dense_1'].data[0][2:23]=0

net.forward(start='output')

rt_output = np.zeros(output_shape)
rt_output = net.blobs['output'].data

print 'output after bit-flip: ', rt_output

print 'MEMORY_ERRORS:'
print "simulating bit-flip in the parameters of layer 'conv_layer_1':"

net.params['conv_layer_1'][1].data[:] = 0

net.forward()

mem_output = np.zeros(output_shape)
mem_output = net.blobs['output'].data

print 'output after bit-flip: ', mem_output


print '** CAUTION: THIS IS A TEST SCENARIO AND CHANGES ARE ARBITRARY. RESULTS AFTER BIT-FLIPS SHOULD NOT BE USED FOR CONCLUSIONS'

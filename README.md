
1. Intro

A modified version of Caffe is proposed with the introduction of a new layer: Injection Layer. It consists on flipping a bit of the input with a probability of 10%.

2. Usage

Compile the caffe distro included in this repo using default instructions (GPU/CPU/OpenCV ...) from the bvlc repository.

In order to use the injection layer with python
export PYTHONPATH=/home/cypox/phd/polsl_uvhc/caffe-net/caffe/python:$PYTHONPATH

Execute "cpp-inference/test_injection.sh" and inspect the "test_results" output file. This test consists on executing a 2-layer network. The first input layer is a 100 neuron layer connected to the injection layer. We forward a vector of all ones and see the effect of the injection layer. The output should be similar to:

1 1
1 1
1 1
1 1
1 1
1 1.00006
1 1
1 1
1 1.00009
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 2.1684e-19
1 1
1 1.88079e-37
1 1
1 1
1 3.00927e-36
1 1
1 1
1 1
1 1
1 1
1 5.04871e-29
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 0.519531
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1.00009
1 1
1 1
1 1
1 1
1 0.523438
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 0.523438
1 1
1 1
1 0.503906
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1
1 1

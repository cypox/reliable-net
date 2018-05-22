CC = g++


#INCLUDE_DIR = /opt/Caffe/caffe/include
#LIBRARY_DIR = /opt/Caffe/caffe/build/lib

INCLUDE_DIR = /home/cypox/phd/polsl_uvhc/caffe-net/caffe/include
LIBRARY_DIR = /home/cypox/phd/polsl_uvhc/caffe-net/caffe/build/lib

#CFLAGS = -std=c++11 -DCPU_ONLY -DUSE_OPENCV
CFLAGS = -std=c++11 -DCPU_ONLY

#LIBS = `pkg-config --libs opencv` -lboost_system -lglog -lcaffe
LIBS = -lboost_system -lglog -lcaffe

forward: forward.cpp
	$(CC) -o $@ $^ -I$(INCLUDE_DIR) $(CFLAGS) -L$(LIBRARY_DIR) $(LIBS)

clean:
	rm -f forward


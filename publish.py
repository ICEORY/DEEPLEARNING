########################################################################
#deep-learning algrithm based on googlenet
#author : ICEORY , SCUT
#date : 2015.8.10
#version : 1.0
########################################################################


########################################################################
#import some dependent modules 
import numpy as np
import os
#set the system path
os.chdir('/home/iceory/iceory/caffe-master/')

import sys
print sys.path
#set the caffe path and insert it into the system path
sys.path.insert(0,  './python')

#this module is the most important module of this program
#you need to set the correct path of this module before import it 
#the module path is /home/iceory/iceory/caffe-master/python
import caffe
########################################################################


########################################################################
#define a GoogleNet class 
#main functions:
#		 __init__(self): initialize the whole network
#		 detectData(self, imgData): key function to detect the image data 
#main parameters:
#		out: save the index of the max predicted result ,0: human, 1: others, 2:car
#		result: the predicted result with the format of [ human, other, car], and sum = 1 
class GoogleNet:
	def __init__(self):
		caffe.set_device(0)
		caffe.set_mode_gpu()
		
		#load the trained network models 
		self.net = caffe.Net('models/bvlc_googlenet/deploy.prototxt',
				'models/bvlc_googlenet/google_iters500_car_model_1.caffemodel',
				caffe.TEST)

		self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		self.transformer.set_transpose('data', (2, 0, 1))
		self.transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
		self.transformer.set_raw_scale('data', 255)
		self.transformer.set_channel_swap('data', (2, 1, 0))
		self.net.blobs['data'].reshape(10, 3, 224, 224)

	def detectData(self, imgData):
		self.net.blobs['data'].data[...] = self.transformer.preprocess('data', caffe.io.load_image(imgData))
		self.out = self.net.forward()
		self.top = self.net.blobs['prob'].data[0].flatten().argsort()[-1]
		self.result = self.net.blobs['prob'].data[0]
########################################################################


########################################################################
#write testing code here to test the network
'''testNet = GoogleNet()
testNet.detectData('/home/iceory/iceory/caffe-master/examples/images/human_1.bmp')
print testNet.top 
print testNet.result
'''
########################################################################

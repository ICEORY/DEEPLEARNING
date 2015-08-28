###import some nessecery modules### 
import os
###set the root path###
os.chdir('/home/iceory/iceory/caffe-master/')

import sys
###set the  path of  caffe-python ###
sys.path.insert(0, './python')

import caffe
import numpy as np

###import module for plot a image###
from pylab import *

####set the iters of  network_trianing###
niter = 250

###nitialize the train_loss matrix for saving datas###
train_loss = np.zeros(niter)

###set the mode of  caffe to GPU_MODE so as to improve the speed of compute###

#caffe.set_device(0)
#caffe.set_mode_gpu()

###create a network according to  the file 'solver.prototxt ' and the script of  'train_val.prototxt'###
solver = caffe.SGDSolver('models/bvlc_googlenet/solver.prototxt')
solver.net.copy_from('models/bvlc_googlenet/google_iters200_car_model_1.caffemodel')

###begin to train the network###

for it in range(niter):
	solver.step(1)
	train_loss[it] = solver.net.blobs['loss'].data
	print 'iter %d, googlenet_loss = %f' % (it, train_loss[it])
print 'done'
solver.net.save('models/bvlc_googlenet/google_iters500_car_model_1.caffemodel')
plot(np.vstack(train_loss))

#caffe.set_mode_cpu()
test_iters = 10
accuracy = 0
for it in range(test_iters):
	solver.test_nets[0].forward()
	accuracy += solver.test_nets[0].blobs['accuracy'].data
accuracy/= test_iters

print 'Accuracy of GoogleNet is:%f' % accuracy
print 'done'
show()

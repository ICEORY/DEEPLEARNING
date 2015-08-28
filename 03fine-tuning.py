import os
os.chdir('/home/iceory/iceory/caffe-master/')
import sys
sys.path.insert(0, './python')

import caffe
import numpy as np
#from pylab import *

niter = 50

train_loss = np.zeros(niter)
scratch_train_loss = np.zeros(niter)

#caffe.set_device(0)
#caffe.set_mode_gpu()

solver = caffe.SGDSolver('models/finetune_flickr_style/solver.prototxt')
#solver.net.copy_from('models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
#scratch_solver = caffe.SGDSolver('models/finetune_flickr_style/solver.prototxt')

for it in range(niter):
	solver.step(1)
#	scratch_solver.step(1)
	train_loss[it] = solver.net.blobs['loss'].data
#	scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data
	#if it % 10 == 0:
	print 'iter %d : loss = %f' % (it, train_loss[it])
		#print 'iter %d, googlenet_loss=%f, scratch_loss=%f' % (it, train_loss[it], scratch_train_loss[it])

print 'done: finish training'

test_iters = 10
accuracy = 0
#scratch_accuracy = 0

for it in range(test_iters):
	solver.test_nets[0].forward()
	accuracy += solver.test_nets[0].blobs['accuracy'].data
#	scratch_solver.test_nets[0].forward()
#	scratch_accuracy += scratch_solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters
#scratch_accuracy /= test_iters
print 'Accuracy for flickr_net:', accuracy
#print 'Accuracy for training from scratch:', scratch_accuracy
print 'done: finish test'

import numpy as np
import os
os.chdir('/home/iceory/iceory/caffe-master/')

import sys
sys.path.insert(0,  './python')

import caffe


#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('models/bvlc_googlenet/deploy.prototxt',
	'models/bvlc_googlenet/google_iters200_car_model_1.caffemodel',
	#'result/single/iters20/google_iters_single_model_2.caffemodel',
	caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

net.blobs['data'].reshape(10, 3, 224, 224)
for number in range(7):
	path = 'examples/images/human_{}'.format(number + 1)
	net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(path + '.bmp'))

	out = net.forward()
	print '------------------------------------------------------------------------------------------------------------------'
	print '------------------------------------------------------------------------------------------------------------------'
	print number + 1
	print("Predicted class is #{}.".format(out['prob'].argmax()))

	labels = np.loadtxt('data/real_car/synwords.txt', str, delimiter='\t')
	top_k = net.blobs['prob'].data[0].flatten().argsort()[-1]
	print net.blobs['prob'].data[0]
	print labels[top_k]
	print 'done'
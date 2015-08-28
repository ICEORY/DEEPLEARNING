
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('/home/iceory/iceory/caffe-master')

import sys
sys.path.insert(0, './python')

import caffe

import h5py
import shutil
import tempfile

import sklearn
import sklearn.datasets
import sklearn.linear_model

x, y = sklearn.datasets.make_classification(
	n_samples=10000, n_features=4, n_redundant=0, n_informative=2,
	n_clusters_per_class=2, hypercube=False , random_state=0
	)

x, xt, y, yt = sklearn.cross_validation.train_test_split(x, y)

ind = np.random.permutation(x.shape[0])[:1000]

clf = sklearn.linear_model.SGDClassifier(
	loss='log', n_iter=1000, penalty='l2', alpha=1e-3, class_weight='auto')
clf.fit(x, y)
yt_pred = clf.predict(xt)
print('Accuracy:{:.3f}'.format(sklearn.metrics.accuracy_score(yt, yt_pred)))

#this file is assumed to be caffe_root/examples/hd5_classification.ipynb
dirname = os.path.abspath('./examples/hd5_classification/data')
if not os.path.exists(dirname):
	os.makedirs(dirname)

train_file_name = os.path.join(dirname, 'trian.h5')
test_file_name = os.path.join(dirname, 'test.h5')

with h5py.File(train_file_name, 'w') as f:
	f['data'] = x
	f['label'] = y.astype(np.float32)

with open(os.path.join(dirname, 'train.txt'), 'w') as f:
	f.write(train_file_name + '\n')
	f.write(test_file_name + '\n')

comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}

with h5py.File(test_file_name, 'w') as f:
	f.create_dataset('data', data=xt, **comp_kwargs)
	f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
with open(os.path.join(dirname, 'test.txt'), 'w') as f:
	f.write(test_file_name + '\n')

from caffe import layers as L
from caffe import params as P

################################################################

def logreg(hdf5, batch_size):
	#logistic regression:data, matrix multiplication,and 2-class softmax loss
	n = caffe.NetSpec()
	n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
	n.ip1 = L.InnerProduct(n.data, num_output=2, weight_filler=dict(type='xavier'))
	n.accuracy = L.Accuracy(n.ip1, n.label)
	n.loss = L.SoftmaxWithLoss(n.ip1, n.label)
	return n.to_proto()

with open('examples/hd5_classification/logreg_auto_train.prototxt', 'w') as f:
	f.write(str(logreg('examples/hd5_classification/data/train.txt', 10)))

with open('examples/hdf5_classification/logreg_auto_test.prototxt', 'w') as f:
	f.write(str(logreg('examples/hdf5_classification/data/test.txt', 10)))

caffe.set_mode_cpu()
solver = caffe.get_solver('examples/hdf5_classification/solver.prototxt')
solver.solve()

accuracy = 0
batch_size = solver.test_nets[0].blobs['data'].num
test_iters = int(len(xt) / batch_size)


for i in range(test_iters):
	solver.test_nets[0].forward()
	accuracy += solver.test_nets[0].blobs['accuracy'].data

accuracy /= test_iters

print ("Accuracy:{:.3f}".format(accuracy))
##################################################################

def nonlinear_net(hdf5, batch_size):
#one  small nonlinearnet , on liap for model kind
	n = caffe.NetSpec()
	n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
	n.ip1 = L.InnerProduct(n.data, num_output=40, weight_filler=dict(type='xavier'))
	n.relu1 = L.ReLU(n.ip1, in_place=True)
	n.ip2 = L.InnerProduct(n.ip1, num_output=2, weight_filler=dict(type='xavier'))
	n.accuracy = L.Accuracy(n.ip2, n.label)
	n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
	return n.to_proto()

with open('examples/hdf5_classification_auto_train.prototxt', 'w') as f:
	f.write(str(nonlinear_net('exampless/hdf5_classification/data/train.txt', 10)))
with open('examples/hdf5_classification/nonlinear_auto_test.prototxt', 'w') as f:
	f.write(str(nonlinear_net('examples/hdf5_classification/data/test.txt', 10)))

solver = caffe.get_solver('examples/hdf5_classification/nonlinear_solver.prototxt')
solver.solve()

accuracy = 0
batch_size = solver.test_nets[0].blobs['data'].num
test_iters = int(len(xt) / batch_size)

for i in range(test_iters):
	solver.test_nets[0].forward()
	accuracy += solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters
print('Accuracy:{:.3f}'.format(accuracy))

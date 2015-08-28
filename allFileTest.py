import numpy as np
import os
os.chdir('/home/iceory/iceory/caffe-master/')

import sys
sys.path.insert(0,  './python')

import caffe

from pylab import *
import os
from Tkinter import *

#caffe.set_device(0)
#caffe.set_mode_gpu()
net = caffe.Net('models/bvlc_googlenet/deploy.prototxt',
	'models/bvlc_googlenet/google_iters500_car_model_1.caffemodel',
	caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

net.blobs['data'].reshape(10, 3, 224, 224)
labels = np.loadtxt('data/real_car/synwords.txt', str, delimiter='\t')
#####################################################
file_path = '/home/iceory/iceory/caffe-master/data/newsavepic_gif/'
temp_name = os.listdir(file_path)
file_list = []
for fn in temp_name:
	file_name = os.path.join(file_path, fn)
	file_list.append(file_name)
file_list.sort()
#print file_list

####################################################
root = Tk()
canvas = Canvas(root, width=300, height=400, bg='white')
img = PhotoImage(file=file_list[0])
canvas_img = canvas.create_image(150,  250, image=img)
canvas_title = canvas.create_text(110, 30, text='Date: 20150809\n Test: googlenet_iters500_models',fill='black')
canvas_t1 = canvas.create_text(125, 60, text='waiting', fill='red')
canvas_t2 = canvas.create_text(150, 80, text=' ', fill='red')
canvas.pack()

####################################################
result = np.ones(3)
pre_human = np.zeros(5)
pre_others = np.zeros(5)
pre_car = np.zeros(5)
final_result = np.ones(5)
####################################################

for i in range(10):
	net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(file_list[i]))
	out = net.forward()

	##freshing the result
	pre_human[0:4] = pre_human[1:5]
	pre_others[0:4] = pre_others[1:5]
	pre_car[0:4] = pre_car[1:5]

	pre_human[4] = net.blobs['prob'].data[0][0]
	pre_others[4] = net.blobs['prob'].data[0][1]
	pre_car[4] = net.blobs['prob'].data[0][2]

	result[0] = np.mean(pre_human)
	result[1] = np.mean(pre_others)
	result[2] = np.mean(pre_car)

	print 'mean of five test is :' + str(result)

	final_result[0:4] = final_result[1:5]
	final_result[4] = result.argsort()[-1]

	##show the flag on the picture
	print 'picture %05d : '%i + str(net.blobs['prob'].data[0])
	if  i >= 5:
		img = PhotoImage(file=file_list[i])
		print '\t result is:  '+str(labels[final_result[4]])
		text = 'PIC %05d  ' %i + 'prediction:' + str(labels[final_result[4]])  + '\nresults of 5 pictures : {}'.format(final_result)
		canvas.itemconfig(canvas_img, image=img)
		canvas.itemconfig(canvas_t1, text=text)
		canvas.itemconfig(canvas_t2, text='prob: ' + format(net.blobs['prob'].data[0]))
		root.update()
mainloop()

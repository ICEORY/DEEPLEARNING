#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

with open('/home/iceory/iceory/caffe-master/data/ilsvrc12/det_synset_words.txt') as f:
	labels_df = pd.DataFrame([
		{
			'synset_id': l.strip().split(' ')[0],
			'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
		}
		for l in f.readlines()
		])
print labels_df
labels_df.sort('name')
print labels_df
'''
def nms_detections(dets, overlap=0.3):
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	ind = np.argsort(dets[:, 4])

	w = x2 - x1
	h = y2 - y1
	area = (w * h).astype(float)

	pick = []
	while len(ind) >0:
		i = ind[-1]
		pick.append(i)
		ind = ind[:-1]

		xx1 = np.maximum(x1[i], x1[ind])
		yy1 = np.maximum(y1[i], y1[ind])
		xx2 = np.maximum(x2[i], x2[ind])
		yy2 = np.maximum(y2[i], y2[ind])

		w = np.maximum(0., xx2 - xx1)
		h = np.maximum(0., yy2 - yy1)

		wh = w * h
		o = wh / (area[i] + area[ind] - wh)
		ind = ind[np.nonzero(o <= overlap)[0]]
	return det[pick, :]
'''
import os
import easygui 
file_list = []
find_path =  easygui.diropenbox(msg='please choose a path', title='path', default='/home/iceory/iceory/caffe-master/data/real_car/')
if find_path != None:	
	file_names = os.listdir(find_path)
	for fn in file_names:
		fullfilename = os.path.join(find_path, fn)
		file_list.append(fullfilename)
	with open(find_path+ 'data.txt', 'w') as f:
		for i in range(len(file_list)):
			f.write(str(file_list[i] + '\t' + '2' + '\n'))
	print 'done'
else:
	print 'you choose nothing'
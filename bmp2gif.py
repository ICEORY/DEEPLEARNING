import os
import easygui 
import Image

file_list = []
find_path =  easygui.diropenbox(msg='please choose a path', title='path', default='/home/iceory/iceory/caffe-master/data/')
if find_path != None:	
	file_names = os.listdir(find_path)
	os.mkdir(find_path + '_gif/')
	for fn in file_names:
		fullfilename = os.path.join(find_path, fn)
		file_list.append(fullfilename)
	file_list.sort()
	for i in range(len(file_list)):
		gif_name = find_path + '_gif/%05d.gif'%i
		#print gif_name
		im = Image.open(file_list[i])
		im.save(gif_name)
	print 'done'
else:
	print 'you choose nothing'
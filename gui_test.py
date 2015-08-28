import numpy as np 
import mayavi.mlab as mlab 
import moviepy.editor as mpy 

duration = 2

fig = mlab.figure(size=(500,500),bgcolor=(1,1,1))
u = np.linspace(0,2*np.pi, 100)
xx, yy, zz = np.cos(u), np.sin(3*u), np.sin(u)
l = mlab.plot3d(xx, yy, zz, representation="wireframe", tube_sides=5, line_width=.5, tube_radius=0.2, figure=fig)

def make_frame(t):
	y = np.sin(3*u)*(0.2 + 0.5*np.cos(2*np.pi*t/duration))
	l.mlab_source.set(y=y)
	mlab.view(azimuth=360*t/duration, distance=9)
	return mlab.screenshot(antialiased=True)

animation = mpy.VideoClip(make_frame, duration=duration).resize(0.5)
animation.write_videofile("wireframe.mp4", fps=20)
animation.write_gif("wireframe.gif", fps=20)
import numpy as np
import matplotlib.pyplot as plt
import os ,cv2

# file_name = cv2.imread(os.path.join(os.getcwd() ,'pred_mt.jpg'))
file_name = cv2.imread(os.path.join(os.getcwd() ,'pred_size.jpg'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(file_name)
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(int(ix) ,',', int(iy))
    global coords
    coords = [int(ix) , int(iy)]
    return coords
for i in range(0,1):
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
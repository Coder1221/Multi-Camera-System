import numpy as np
import cv2,os
import matplotlib.pyplot as plt
import pandas as pd
pdd = os.path.join(os.getcwd(), '3c.txt')
pdd1 = os.path.join(os.getcwd(), '3c_P.txt')

df = pd.read_csv(pdd,header=None)
df1 = pd.read_csv(pdd1,header=None)

#sk 1
# muq 2
# mt 3
image1 = cv2.imread(os.path.join(os.getcwd() , 'pred_mt.jpg'))
image2 = cv2.imread(os.path.join(os.getcwd() , 'pred_size.jpg'))

img_points  = df.to_numpy()
target_points = df1.to_numpy()

H = cv2.findHomography(img_points,target_points)[0]
print(H)
new_image = cv2.warpPerspective(image1,H,(image1.shape[1], image1.shape[0]),flags=cv2.INTER_LINEAR)
cv2.imwrite(os.path.join(os.getcwd(),'baQwassS.jpg'), new_image)
print('Done')
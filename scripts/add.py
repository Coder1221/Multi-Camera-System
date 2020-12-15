import os,cv2
import numpy as np
a = cv2.imread('/home/anti_super/Downloads/cv/project/ip_cam/seeeeee_1muq.jpg')
b = cv2.imread('/home/anti_super/Downloads/cv/project/ip_cam/seeeeee_1sk.jpg')
c = cv2.imread('/home/anti_super/Downloads/cv/project/ip_cam/seeeeee_mt.jpg')
c = cv2.flip(c, 1)
d = b
temp = np.zeros(shape=(416,416,3),dtype=np.uint8)

# print(temp[0][23].shape)

# for 
# count=0;
# for i ,j, k in zip(a,b,c):
#     for x in range(416):
#         i[x]+j[x]+k[x]        
#         temp[count] =
      
a = np.where(a<=1,0.0,a)
da =(a+b+c)/3

# temp = np.zeros(shape=(416,416,3),dtype=np.uint8)
# da =cv2.hconcat([cv2.flip(c, 1),a])
# db =cv2.hconcat([temp,d])
# D = cv2.vconcat([da,db])
cv2.imwrite(os.path.join(os.getcwd(),'flipeavg.jpg') , (c+a + b)/3)
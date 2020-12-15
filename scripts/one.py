import cv2 ,os
filename= os.path.join(os.getcwd() , 'mt.mp4')
vs = cv2.VideoCapture(filename)
frameSize= (416,416)
while True:
    (grabbed, frame) = vs.read()        
    if not grabbed:
        break
    frame = cv2.resize(frame , frameSize)
    cv2.imwrite(os.path.join(os.getcwd(),'pred_mt.jpg'), frame)
    break

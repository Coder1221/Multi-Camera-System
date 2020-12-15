import cv2 , datetime ,os
import matplotlib.pyplot as plt
from utils import *
from darknet import Darknet
import argparse
import pandas as pd
from centroidtracker import CentroidTracker
import numpy as np,imutils, dlib


parser = argparse.ArgumentParser(description='IP-web Cam for yolo')
#  different options for program
parser.add_argument('--live' , action="store_true" ,help='Put scrip in live mood')
parser.add_argument('--recorded' , action="store_true" ,help='Put video in recording folder' )
parser.add_argument('--projected' ,action="store_true",help='vedio and their coordiantes')
parser.add_argument('--HeatMap' ,action="store_true",help='Generate heat map')
parser.add_argument('--Animated_HeatMap' ,action="store_true",help='Generate  animated heat map for k steps')
parser.add_argument('--Bonus' ,action="store_true" , help = "Implemented Car Counting" )

# change file name here
parser.add_argument('--filename' , type= str,default='Projected.avi' , help= 'Provide file name')
# settings for model
parser.add_argument('--width' , type = int,default=416 , help= 'Width' )
parser.add_argument('--height' , type = int,default=416 , help= 'height')
parser.add_argument('--detect' , type = int ,default=1)
parser.add_argument('--Record', action="store_true")
args = parser.parse_args()

file_name = os.path.join(os.getcwd() , 'recordings' , args.filename)

all_cameras = False
width  = args.width
height = args.height
frameSize = (args.width,args.height)
heat_map_dict ={}


k_animated = 10 # set k here for animated k steps
k = 21
gauss = cv2.getGaussianKernel(k, np.sqrt(64))
gauss = gauss * gauss.T # 2d gaussian window or kernal
gauss = (gauss/gauss[int(k/2),int(k/2)])


all_points_heat_map = []

if args.detect:
    cfg_file = './cfg/yolov3.cfg'
    weight_file = './weights/yolov3.weights'
    namesfile = 'data/coco.names'
    m = Darknet(cfg_file)
    print('\nLoading weights of Darknet\n')
    m.load_weights(weight_file)
    class_names = load_class_names(namesfile)
    print('\n Done loading weights\n')


def Detect(frame , circle = False , bonding_box = False , homography = None):
    temp  = frame.copy()
    points = []    
    resized_frame  = cv2.resize(temp , (m.width ,m.height))
    iou_thresh = 0.4
    nms_thresh = 0.6
    boxes = detect_objects(m,resized_frame ,iou_thresh , nms_thresh)
    for i in boxes:        
        color = (255, 0, 0)
        thickness = 2
        x1 = int(np.around((i[0] - i[2]/2.0) * width))
        y1 = int(np.around((i[1] - i[3]/2.0) * height))
        x2 = int(np.around((i[0] + i[2]/2.0) * width))
        y2 = int(np.around((i[1] + i[3]/2.0) * height))
        end =  (x2,y2)
        start = (x1,y1)
        mid =  int((x1+x2)*0.5)
        mid2 = int((y1+y2)*0.5)
        ind = int(i[6].item())
        class_name  = class_names[ind]
        
        if circle:
            points.append((mid,y2))


        if class_name in ['person' ,'car'] and bonding_box:
            if class_name =='person':
                color = (0,0,255)
            else:
                color = (0,255,0)
            cv2.rectangle(temp,end, start, color,thickness)            
            cv2.putText(temp,class_name, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return temp , points

if args.Record:
    cwd = os.getcwd()
    folder = os.path.join(cwd,'recordings')
    #  directory struture
    if os.path.isdir(folder):
	    print('folder exits')
    else:
    	os.mkdir(folder)

temp_frame = 0

if args.live:
    print("\nPress escape to exit camera streaming and recording as well\n")
    x = datetime.datetime.now()
    cam1 = cv2.VideoCapture(0)
   
   
    if all_cameras:
        filename2 = 'recordings/cam2-'+str(x)+'.avi'
        record2 = cv2.VideoWriter(filename2, cv2.VideoWriter_fourcc(*'MJPG') , 10 , frameSize)
        filename3 = 'recordings/cam3-'+str(x)+'.avi'
        record3 = cv2.VideoWriter(filename3, cv2.VideoWriter_fourcc(*'MJPG') , 10 , frameSize)
        
    filename = 'recordings/cam1-'+str(x)+'.avi'
    record = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MJPG') , 10 , frameSize)
    
    if all_cameras:
        address = None
        cam2 = cv2.VideoCapture(address)
        address3 = None
        cam3 = cv2.VideoCapture(address3)
    
    if (cam1.isOpened()== False): # all cameras check
        print("Error opening video stream or file")
    
    while(cam1.isOpened()): # all cameras check
        ret, frame = cam1.read()
        if all_cameras:
            ret1, frame1 = cam2.read()
            ret2 ,frame2 = cam3.read()
            
        frame = cv2.resize(frame, frameSize)
        if args.detect:
            frame = Detect(frame,0,1)
        
        if args.Record:
            record.write(cv2.resize(frame,frameSize))
            if all_cameras:
                record2.write(cv2.resize(frame1,frameSize))
                record3.write(cv2.resize(frame2,frameSize))        
        
        
        if ret==True:
            if all_cameras:
                # projected view ADN COMBINED VIEW OF ALL THREE CAMERAS
                c = homo(Detect(frame  ,1 ,0) ,'3.txt' , '3_P.txt')
                a = homo(Detect(frame1 ,1 ,0) ,'M_2.txt' , 'M_P2.txt')
                b = homo(Detect(frame2 ,1 ,0) ,'M_1.txt' , 'M_1P.txt')

                final =(a/255 +b/255 + c/255)*255
                final = (final/3)*2
                final = np.clip(final, a_min = 0.0, a_max = 255.0)
                final = final.astype(np.uint8)
        
                da = cv2.hconcat([final , Detect(frame2,0,1)])
                db = cv2.hconcat([Detect(frame1,0,1) ,Detect(frame,0,1)])
                frame =  cv2.vconcat([da,db])
                cv2.imshow('Frame',frame)  # show all views      
            else:
                cv2.imshow('Frame',frame)
                
            if (cv2.waitKey(1) & 0xFF == 27):
                print('\nRecoding saved in recodings folder')
                break
        else:
            break
    cam1.release()
    if all_cameras:
        cam2.release()
        cam3.relesee()

if args.recorded:
    print('\n Started Detecting objects in video  in file \n')
    
    vs = cv2.VideoCapture(file_name)
    writer = None
    while True:
        temp_frame+=1
        (grabbed, frame) = vs.read()        
        if not grabbed:
            break
        frame = cv2.resize(frame , frameSize)
      
        if writer is None:
            pred= 'Pred_' + args.filename
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(pred, fourcc, 10, (frame.shape[1], frame.shape[0]), True)
        
        if args.detect  and temp_frame % 5 == 0:
            frame = Detect(frame, 0,1)

        writer.write(frame)
    writer.release()
    vs.release()
    print('\nDone with the video')


def homo(image1, points , p_points):
    pdd =  os.path.join(os.getcwd(), 'video' ,'two' , points)
    pdd1 = os.path.join(os.getcwd(), 'video' , 'two' , p_points)
    df = pd.read_csv(pdd,header=None)
    df1 = pd.read_csv(pdd1,header=None)
    img_points  = df.to_numpy()
    target_points = df1.to_numpy()
    H = cv2.findHomography(img_points,target_points)[0]
    new_image = cv2.warpPerspective(image1,H,(image1.shape[1], image1.shape[0]),flags=cv2.INTER_LINEAR)
    return new_image


def put_circle(frame , points  ,homography ,circle = True ):
    heat = []
    for i in points:
        z,x = i
        mapped = np.ones((3, 1), dtype = np.float32)
        mapped[0][0] = z
        mapped[1][0] = x
        pt = homography @ mapped                
        pt = pt / pt[2][0]
        x = pt[0][0]
        y = pt[1][0]
        peepy = (int(x),int(y))
        heat.append(peepy)
        if circle:
            cv2.circle(frame ,peepy, 5, (0,255,0), -1)
    return frame ,heat

def Homo_point(image1, points , p_points):
    pdd =  os.path.join(os.getcwd(), 'video' ,'two' , points)
    pdd1 = os.path.join(os.getcwd(), 'video' , 'two' , p_points)
    df = pd.read_csv(pdd,header=None)
    df1 = pd.read_csv(pdd1,header=None)
    img_points  = df.to_numpy()
    target_points = df1.to_numpy()
    H = cv2.findHomography(img_points,target_points)[0]
    new_image = cv2.warpPerspective(image1,H,(image1.shape[1], image1.shape[0]),flags=cv2.INTER_LINEAR)
    return new_image, H

def heatmap(frame, data):
    frame = frame/255.0
    img2 = np.zeros((frame.shape[0],frame.shape[1],3)).astype(np.float32)
    j = cv2.cvtColor(cv2.applyColorMap(((gauss)*255).astype(np.uint8) ,cv2.COLORMAP_AUTUMN),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    k = 21

    for p in data:
        x,y = p
        if x>=frame.shape[1]-11 or y>=frame.shape[1]-11:
            continue
        b = img2[x-int(k/2):x+int(k/2)+1 ,y-int(k/2):y+int(k/2)+1 ,:]
        c = b + j
        img2[x-int(k/2):x+int(k/2)+1 ,y-int(k/2):y+int(k/2)+1 ,:] = c

    m = np.max(img2 ,axis=(0,1)) + 0.0001
    img2 = img2/m
    
    g = cv2.cvtColor(img2 ,cv2.COLOR_RGB2GRAY)
    mask = np.where(g>0.2,1,0).astype(np.float32)
    
    mask_3 = np.ones((frame.shape[0], frame.shape[1],3))*(1-mask)[:,:,None]
    mask_4 = img2 * (mask)[:,:,None]
    
    new_top = mask_3 * frame
    Heatmap = new_top + mask_4
    return Heatmap*255

def detect_C(frame):
	boxes = []
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (416, 416), 127.5)
	net.setInput(blob)
	detections = net.forward()
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.4:
			idx = int(detections[0, 0, i, 1])
			if idx!= 7:
				continue
			box = detections[0, 0, i, 3:7] * np.array([416, 416, 416, 416]) # scaling back to image coordinates
			(startX, startY, endX, endY) = box.astype("int")    
			boxes.append((startX, startY, endX, endY))
	return boxes

def TRACKER(trackers ,rgb , frame ,rects): # function for updating each tracker
	for tracker in trackers: 
		tracker.update(rgb) # update position of frame and replot on the frame
		pos = tracker.get_position()
		startX = int(pos.left())
		startY = int(pos.top())
		endX = int(pos.right())
		endY = int(pos.bottom())
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)
		rects.append((startX, startY, endX, endY))
	return rects

class TrackableObject:  # a object in frame
	def __init__(self, objectID, centroid):
		self.objectID = objectID # unique id of each object
		self.centroids = [centroid]
		self.counted = False


if args.Bonus:
    net = cv2.dnn.readNetFromCaffe( "MODEL/SSD_MODEL.txt", "MODEL/SSD_MODEL.bin") # loading model
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50) # centroid tracker for tracking bounding box (works on euclidean distance to compute distance between centroid)
    trackableObjects = {}
    total = 0
    totalFrames = 0

    vs = cv2.VideoCapture("video/sk.mp4")
    cap1 = cv2.VideoCapture("video/mt.mp4")
    cap2 = cv2.VideoCapture("video/muqi.mp4")
    writer = None
    while True:
        (grabbed, frame) = cap1.read()        
        (grabbed1, frame1) = cap2.read()        
        (grabbed2, frame2) = vs.read()
        if not grabbed or not grabbed2 or not grabbed1:
            break
        
        frame = cv2.resize(frame , frameSize)
        frame1 = cv2.resize(frame1 , frameSize)
        frame2 = cv2.resize(frame2 , frameSize)
        frame2_temp = frame2.copy() 
        rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        rects = []
        if totalFrames % 5 == 0: # skipping 5 frame per seconds
            trackers = []
            b = detect_C(frame2) # return bounding boxes
            for i in b:
                startX, startY, endX, endY = i
                tracker = dlib.correlation_tracker()#This is a tool for tracking moving objects in a video stream. 
                #You give it the bounding box of an object in the first frame and it attempts to track the object in the box from frame to frame. 
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)  # start trackign of object
                trackers.append(tracker)
        else:
            rects = TRACKER(trackers ,rgb , frame2 ,rects) # keep on updating frame for each frame
        
        objects = ct.update(rects)
        for (objectID, centroid) in objects.items(): 
            to = trackableObjects.get(objectID, None)
            if to is None: # if we have new object id
                to = TrackableObject(objectID, centroid) #registing new object
            else:
                if to.counted==False: # and if that object is not counted then add the counter to 1
                    total += 1
                    to.counted = True
            trackableObjects[objectID] = to # adding that object to dictonary


        #----------------------------
        a ,_ = Homo_point(frame2_temp,'M_1.txt' , 'M_1P.txt')
        b ,_ = Homo_point(frame1,'M_2.txt' , 'M_P2.txt')
        c ,_ = Homo_point(frame,'3.txt' , '3_P.txt')
        
        strr = "Car counter => {}".format(total)
        # creating polygon on image and placing text between 
        penta = np.array([[[40,160],[120,100],[200,160],[160,240],[80,240]]], np.int32)
        cv2.putText(frame2, strr, (50,170),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.polylines(frame2, [penta], True, (255,120,255),3)
                                
        final = (a/255 +b/255 + c/255)*255
        final = (final/3)*2
        final = np.clip(final, a_min = 0.0, a_max = 255.0)
        final = final.astype(np.uint8)
        
        da = cv2.hconcat([final , frame2])
        db = cv2.hconcat([frame1 ,frame])
        D =  cv2.vconcat([da,db])
        
        cv2.imshow('Frame' ,D)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if writer is None:
            pred = 'Bonus.mp4'
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(pred, fourcc, 10, (416*2, 416*2), True)
        writer.write(D)
        totalFrames += 1
    writer.release()
    vs.release()
    cv2.destroyAllWindows()

if args.projected:
    DIR = os.path.join(os.getcwd() , 'video')
    vid1 = os.path.join(DIR ,'mt.mp4')
    vid2 = os.path.join(DIR ,'muqi.mp4')
    vid3 = os.path.join(DIR ,'sk.mp4')
    
    cap0 = cv2.VideoCapture(vid1)
    cap1 = cv2.VideoCapture(vid2)
    cap2 = cv2.VideoCapture(vid3)

    writer = None
    print("\nStarting projection video of 3 views....\n")
    temp_frame =0
    
    heat_map = False
    circle = True
    frame_count = 0
    removed_step = 0 # removal of time step k 

    if args.HeatMap:
        circle = False

    while True:
        (grabbed, frame) = cap0.read()        
        (grabbed1, frame1) = cap1.read()        
        (grabbed2, frame2) = cap2.read()
                
        if not grabbed or not grabbed2 or not grabbed1:
            break
        
        frame = cv2.resize(frame , frameSize)
        frame1 = cv2.resize(frame1 , frameSize)
        frame2 = cv2.resize(frame2 , frameSize)
        
        temp_frame += 1
    
        if temp_frame % 5 == 0: #every fifth frame
            s1 ,H1 = Homo_point(frame2,'M_1.txt' , 'M_1P.txt')
            
            _ ,p1 = Detect(frame2  ,1 ,0)
            b,heat1 = put_circle(s1,p1,H1 ,circle)
            
            
            s2 ,H2 = Homo_point(frame1,'M_2.txt' , 'M_P2.txt')
            _ ,p2 = Detect(frame1  ,1 ,0)
            a,heat2 = put_circle(s2,p2,H2,circle)
            
            s3 ,H3 = Homo_point(frame,'3.txt' , '3_P.txt')
            _ ,p3 = Detect(frame  ,1 ,0)
            c,heat3 = put_circle(s3,p3,H3,circle)
                        
            final = (a/255 +b/255 + c/255)*255
            final = (final/3)*2
            final = np.clip(final, a_min = 0.0, a_max = 255.0)
            final = final.astype(np.uint8)
            
            if args.HeatMap:
                heat_map_points  = heat1 + heat2 +heat3
                heat_map_dict[frame_count] = heat_map_points
                frame_count+=1
                
                if args.Animated_HeatMap and frame_count>k_animated:
                    del heat_map_dict[removed_step]
                    removed_step+=1 # next frame remove the next one
                    ttt = []  # temp for k points
                    for xxx in heat_map_dict.values():
                        for ii in xxx:
                            ttt.append(ii)
                    all_points_heat_map = ttt
                else:
                    all_points_heat_map += heat_map_points
                
                final = heatmap(final, all_points_heat_map)    
                final = final.astype(np.uint8)
                
                    
            da = cv2.hconcat([final , Detect(frame2,0,1)[0] ])
            db = cv2.hconcat([Detect(frame1,0,1)[0] ,Detect(frame,0,1)[0] ])
            D =  cv2.vconcat([da,db])
            
            cv2.imshow('Frame' ,D)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if writer is None:
                pred = 'Video.mp4'
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(pred, fourcc, 10, (416*2, 416*2), True)            
            # writer.write(D)
            # print(temp_frame)
    writer.release()
else:
    if not args.live:
        print('Provide arguements')
        exit()
print('\n Done')
cv2.destroyAllWindows()
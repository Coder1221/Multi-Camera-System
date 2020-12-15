from centroidtracker import CentroidTracker
# from tracker import CentroidTracker
import numpy as np,imutils, dlib ,cv2 ,os
import matplotlib.pyplot as plt
class TrackableObject:
	def __init__(self, objectID, centroid):
		self.objectID = objectID # unique id of each object
		self.centroids = [centroid]
		self.counted = False

net = cv2.dnn.readNetFromCaffe( "MODEL/SSD_MODEL.txt", "MODEL/SSD_MODEL.bin") # loading model
ct = CentroidTracker(maxDisappeared=40, maxDistance=50) # centroid tracker for tracking bounding box (works on euclidean distance to compute distance between centroid)
trackableObjects = {}

def detect(frame):
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

def TRACKER(trackers ,rgb , frame ,rects):
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

vs = cv2.VideoCapture("video/sk.mp4")
total = 0
totalFrames = 0
while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
	  break
	frame = cv2.resize(frame,  (416 ,416))
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rects = []
	if totalFrames % 5 == 0: # skipping 5 frame per seconds
		trackers = []
		b = detect(frame) # return bounding boxes
		for i in b:
			startX, startY, endX, endY = i
			tracker = dlib.correlation_tracker()#This is a tool for tracking moving objects in a video stream. 
			#You give it the bounding box of an object in the first frame and it attempts to track the object in the box from frame to frame. 
			rect = dlib.rectangle(startX, startY, endX, endY)
			tracker.start_track(rgb, rect)  # start trackign of object
			trackers.append(tracker)
	else:
		rects = TRACKER(trackers ,rgb , frame ,rects) # keep on updating frame for each frame
	
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

	strr = "Total cars = {}".format(total)
	penta = np.array([[[40,160],[120,100],[200,160],[160,240],[80,240]]], np.int32)
	cv2.putText(img, strr, (90,170),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.polylines(img, [penta], True, (255,120,255),3)
	
	# cv2.putText(frame, strr, (200,200),	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	totalFrames += 1
vs.release()
cv2.destroyAllWindows()
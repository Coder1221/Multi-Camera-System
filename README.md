# Multi-Camera-surveillance-System
* Projeted view of all three cameras on one plane
* Object detection [car,person] in video by Yolo darknet framework
* Porjection of bounding box on orthograohic view
* Visulization of live heatmap
* Live Animated heat map
* Trip wire (Counting of car) by Centroid tracker and dlib
Download yolo weights and put in Weights folder
# For all projected videos
python3 main.py --projected
# For heat map and animated heat map
python3 main.py --projected --HeatMap --Animated_HeatMap
# For Counting cars in video 
python3 main.py --Bonus

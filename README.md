# Satellite view
![alt text](https://github.com/Coder1221/Multi-Camera-System/blob/master/Satellite%20view.jpeg)
# All views
![alt text](https://github.com/Coder1221/Multi-Camera-System/blob/master/Screenshot%20from%202020-12-19%2007-13-43.png)

# Multi-Camera-surveillance-System
* Projected view of all three cameras on one plane
* Object detection [car,person] in video by Yolo darknet framework
* Projection of bounding box on orthographic view
* Visualization of live heatmap
* Live Animated heat map
* Trip wire (Counting of car) by Centroid tracker and dlib
Download yolo weights and put in Weights folder
# For all projected videos
python3 main.py --projected
# For heat map and animated heat map
python3 main.py --projected --HeatMap --Animated_HeatMap
# For Counting cars in video 
python3 main.py --Bonus

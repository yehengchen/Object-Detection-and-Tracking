<h1 align="center">
  Object Detection and Tracking
</h1>

<div align="center">
  <img src="https://github.com/yehengchen/ObjectDetection/blob/master/img/objectdetection.gif" width="60%" height="60%">
</div>

*Object detection is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos.*

***

## Environment

I have tested on Ubuntu 16.04/18.04. The code may work on other systems.

[[Ubuntu-Deep-Learning-Environment-Setup]](https://github.com/yehengchen/Ubuntu-Deep-Learning-Environment-Setup)

* ##### Ubuntu 16.04 / 18.04 
* ##### ROS Kinetic / Melodic
* ##### GTX 1080Ti / RTX 2080Ti
* ##### python 2.7 / 3.6


## Installation

Clone the repository

```
git clone https://github.com/yehengchen/Object-Detection-and-Tracking.git
```

***
## [OneStage]
### [YOLO](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo): Real-Time Object Detection and Tracking

* __How to train a YOLO model on custom images: YOLOv3 - [[Here]](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3) / YOLOv4 - [[Here]](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo/Train-a-YOLOv4-model)__




***
<img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/output_49.gif" width="60%" height="60%">

* #### YOLOv4 + Deep_SORT - Pedestrian Counting & Social Distance - [[Here]](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo/deep_sort_yolov4)
* #### YOLOv3 + Deep_SORT - Pedestrian&Car Counting - [[Here]](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/deep_sort_yolov3)

***
<img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/sort_1.gif" width="60%" height="60%">

* #### YOLOv3 + SORT - Pedestrian Counting - [[Here]](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3_sort)
***

### [Darknet_ROS](https://github.com/yehengchen/YOLOv3-ROS/tree/master/darknet_ros): Real-Time Object Detection and Grasp Detection

<img src="https://github.com/yehengchen/YOLOv3-ROS/blob/master/darknet_ros/yolo_network_config/weights/output.gif" width="60%" height="60%">

* #### YOLOv3 + ROS Kinetic - For small Custom Data - [[Here]](https://github.com/yehengchen/YOLOv3_ROS)
***

<img src="https://github.com/yehengchen/YOLOv3-ROS/blob/master/yolov3_pytorch_ros/models/output.gif" width="60%" height="100%">

* #### YOLOv3 + ROS Melodic - Robot Grasp Detection - [[Here]](https://github.com/yehengchen/YOLOv3_ROS/tree/master/yolov3_pytorch_ros)

* #### Parts-Arrangement-Robot - [[Here]](https://github.com/yehengchen/Parts-Arrangement-Robot)
***

<img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/chair_pin.gif" width="60%" height="100%">

* #### YOLOv3 + OpenCV + ROS Melodic - Object Detection (Rotated) - [[Here]](https://github.com/yehengchen/YOLOv3-ROS/tree/master/yolov3_grasp_detection_ros)

***
### [DeepLabv3+_ROS](https://arxiv.org/abs/1802.02611): Mars Rover - Real-Time Object Tracking
<img src="https://github.com/HaosUtopia/Mars_Rover/blob/main/deeplabv3plus_ros/imgs/mars_rover_mastcam_rock_tracking.gif" width="60%" height="60%">

* #### DeepLab + OpenCV + ROS Melodic/Gazebo - Object Tracking - [[Here]](https://github.com/HaosUtopia/Mars_Rover/tree/main/deeplabv3plus_ros)

* #### Mars_Rover + ROS Melodic/Gazebo - [[Here]](https://github.com/HaosUtopia/Mars_Rover)

***
### [SSD](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/ssd): Single Shot MultiBox Detector

***
## [TwoStage]
### [R-CNN](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/TwoStage/R-CNN): Region-based methods
*Fast R-CNN / Faster R-CNN / Mask R-CNN*

__How to train a Mask R-CNN model on own images - [[Here]](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/TwoStage/R-CNN)__

<img src="https://github.com/yehengchen/mask_rcnn_ros/blob/master/scripts/mask_rcnn.gif" width="60%" height="60%">

* #### Mask R-CNN + ROS Kinetic - [[Here]](https://github.com/yehengchen/mask_rcnn_ros)

This project is ROS package of Mask R-CNN algorithm for object detection and segmentation.

***

### COCO & VOC Datasets
* #### COCO dataset and Pascal VOC dataset - [[Here]](https://github.com/yehengchen/ObjectDetection/blob/master/COCO%20and%20Pascal%20VOC.md)
* #### How to get it working on the COCO dataset __coco2voc__ - [[Here]](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/coco2voc.md)
* #### Convert Dataset2Yolo - COCO / VOC - [[Here]](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/convert2Yolo)

***

#### CV & Robotics Paper List (3D object detection & 6D pose estimation) - [[Here]](https://github.com/yehengchen/Computer-Vision-and-Robotics-Paper-List)

#### PapersWithCode: Browse > Computer Vision > Object Detection - [[Here]](https://paperswithcode.com/task/object-detection)

#### ObjectDetection Two-stage vs One-stage Detectors - [[Here]](https://github.com/yehengchen/ObjectDetection/blob/master/Two-stage%20vs%20One-stage%20Detectors.md)

#### ObjectDetection mAP & IoU - [[Here]](https://github.com/yehengchen/ObjectDetection/blob/master/mAP%26IoU.md)



*** 



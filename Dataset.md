# COCO and Pascal VOC
![](https://github.com/yehengchen/ObjectDetection/blob/master/img/dataset.png)

#### Browse > Computer Vision > Object Detection - [[Link]](https://paperswithcode.com/task/object-detection)

## COCO
![](https://github.com/yehengchen/ObjectDetection/blob/master/img/coco.png)
![](https://github.com/yehengchen/ObjectDetection/blob/master/img/coco_yolo.png)

### Introduction
*COCO is an image dataset designed to spur object detection research with a focus on detecting objects in context. The annotations include instance segmentations for object belonging to 80 categories, stuff segmentations for 91 categories, keypoint annotations for person instances, and five image captions per image.*

COCO is a large-scale object detection, segmentation, and captioning dataset. 
COCO has several features:
    
    Object segmentation
    Recognition in context
    Superpixel stuff segmentation
    330K images (>200K labeled)
    1.5 million object instances
    80 object categories
    91 stuff categories
    5 captions per image
    250,000 people with keypoints

### COCO2017 Data

* Download the [2017 Train images](http://images.cocodataset.org/zips/train2017.zip) [118K/18GB]
* Download the [2017 Val images](http://images.cocodataset.org/zips/val2017.zip) [5K/1GB]
* Download the [2017 Test images](http://images.cocodataset.org/zips/test2017.zip) [41K/6GB]
* COCO API/[PythonAPI](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI)

## VOC2007
![](https://github.com/yehengchen/ObjectDetection/blob/master/img/voc.png)
![](https://github.com/yehengchen/ObjectDetection/blob/master/img/voc_yolo.png)

### Introduction
*The goal of this challenge is to recognize objects from a number of visual object classes in realistic scenes (i.e. not pre-segmented objects). It is fundamentally a supervised learning learning problem in that a training set of labelled images is provided.*
The twenty object classes that have been selected are:

    Statistics
    20 classes:
    Person: person
    Animal: bird, cat, cow, dog, horse, sheep
    Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
    Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
    Train/validation/test: 9,963 images containing 24,640 annotated objects.
    New developments 
    Number of classes increased from 10 to 20
    Segmentation taster introduced
    Person layout taster introduced
    Truncation flag added to annotations
    Evaluation measure for the classification challenge changed to Average Precision. Previously it had been ROC-AUC.

### VOC2007 Data
The annotated test data for the VOC challenge 2007 is now available: 
* Download the [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (450MB tar file)
* Download the [development kit code and documentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar) (250KB tar file)
* Download the [PDF documentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/devkit_doc_07-Jun-2007.pdf) (120KB PDF) 
* View the [guidelines](View the guidelines used for annotating the database) used for annotating the database
* Download the [annotated test data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (430MB tar file)
* Download the [annotation only](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar) (12MB tar file, no images)


## References
ECCV 2018 Joint COCO and Mapillary Recognition - [[Link]](http://cocodataset.org/workshop/coco-mapillary-eccv-2018.html)

The PASCAL Visual Object Classes Challenge 2007 - [[Link]](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)

Object detection: speed and accuracy comparison - [[Link]](https://medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359)

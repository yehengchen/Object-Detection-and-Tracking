# YOLO: Real-Time Object Detection
*You only look once (YOLO) is a state-of-the-art, real-time object detection system. YOLOv3 On a Titan X it processes images at 40-90 FPS and has a mAP on VOC 2007 of 78.6% and a mAP of 48.1% on COCO test-dev.
YOLOv3 On a Pascal Titan X it processes images at 30 FPS and has a mAP of 57.9% on COCO test-dev.*

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/yologo_1.png" width="40%" height="40%">

| __Model__ | __Train__ |__Test__|__mAP__| __FPS__| __Cfg__| __Weights__|
|-----------| :-------: | :----: | :-----: | :----: | :----: | :--------: |
|YOLOv2 608x608|COCO trainval|test-dev| 48.1 | 40 | [Cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg) | [weights](https://pjreddie.com/media/files/yolov2.weights)||
|Tiny YOLO|COCO trainval|test-dev|23.7|244|[Cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg) | [weights](https://pjreddie.com/media/files/yolov2-tiny.weights)||
|YOLOv3-320|COCO trainval|test-dev| 51.5 | 45 | [Cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) | [weights](https://pjreddie.com/media/files/yolov3.weights)||
|YOLOv3-608|COCO trainval|test-dev| 57.9 | 20 | [Cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) | [weights](https://pjreddie.com/media/files/yolov3.weights)||
|YOLOv3-tiny|COCO trainval|test-dev| 33.1 | 220 | [Cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg) | [weights](https://pjreddie.com/media/files/yolov3-tiny.weights)||

***
# YOLOv1 - Redmon et al., 2016
*You Only Look Once:Unified, Real-Time Object Detection - [[Paper]](https://arxiv.org/pdf/1506.02640.pdf)*

The YOLO model is the very first attempt at building a fast real-time object detector. Because YOLO does not undergo the region proposal step and only predicts over a limited number of bounding boxes, it is able to do inference super fast.
![The YOLO Detection System](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/yolov1.png)

Processing imageswith YOLO is simple and straightforward. 
* (1) resizesthe input image to 448×448.
* (2) runs a single convolutional network on the image.
* (3) thresholds the resulting detections bythe model’s confidence.

## Workflow

Pre-train a CNN network on image classification task.
* The coordinates of bounding box are defined by a tuple of 4 values, (center x-coord, center y-coord, width, height) — (x,y,w,h), where x and y are set to be offset of a cell location. Moreover, x, y, w and h are normalized by the image width and height, and thus all between (0, 1].
* A confidence score indicates the likelihood that the cell contains an object: Pr(containing an object) x IoU(pred, truth); where Pr = probability and IoU = interaction under union

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/yolo.png" width="60%" height="60%">

It divides the image into an S × S grid and for each grid cell predicts B bounding boxes, confidence for those boxes,
and C class probabilities.These predictions are encoded as an __S × S × (B ∗ 5 + C)__ tensor.
For evaluating YOLO on P ASCAL VOC, They use S = 7, B = 2. P ASCAL VOC has 20 labelled classes so C = 20.

__a: the location of B bounding boxes__

__b: Confidence as Pr(Object) ∗ IOU (truth | pred)__

*(b) a confidence score*
    
    If no object exists in that cell, the confidence scores should be zero. (Pr(Object) = 0)
    intersection over union (IOU) between the predicted box and the ground truth.
    
__c: Class probabilities, Pr(Class i | Object)__

*(c) a probability of object class conditioned on the existence of an object in the bounding box*

__Pr(Class i |Object) ∗ Pr(Object) ∗ IOU(truth | pred) = Pr(Class i ) ∗ IOU(truth | pred)__

## Network Architecture

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/yolov1network.png" width="60%" height="60%">
<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/yolo-network-architecture.png" width="60%" height="60%">

*Network has 24 convolutional layers followed by 2 fully connected layers. Alternating 1×1 convolutional layers reduce the features space from preceding layers.*

__The final prediction is a 7 × 7 × 30 tensor.__

## Loss Function
The loss consists of two parts, the localization loss for bounding box offset prediction and the classification loss for conditional class probabilities. Both parts are computed as the sum of squared errors. Two scale parameters are used to control how much we want to increase the loss from bounding box coordinate predictions (λcoord) and how much we want to decrease the loss of confidence score predictions for boxes without objects (λnoobj). Down-weighting the loss contributed by background boxes is important as most of the bounding boxes involve no instance. In the paper, the model sets λcoord=5 and λnoobj=0.5.

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/yolov1_lossfunc.png" width="80%" height="80%">

where,

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/Screenshot%20from%202019-05-18%2016-55-25.png" width="60%" height="60%">


<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/yolo-responsible-predictor.png" width="80%" height="80%">

*At one location, in cell i, the model proposes B bounding box candidates and the one that has highest overlap with the ground truth is the “responsible” predictor.*

The loss function only penalizes classification error if an object is present in that grid cell, 𝟙obji=1. It also only penalizes bounding box coordinate error if that predictor is “responsible” for the ground truth box, 𝟙objij=1.

As a one-stage object detector, YOLO is super fast, but it is not good at recognizing irregularly shaped objects or a group of small objects due to a limited number of bounding box candidates.

***
# YOLOv2
### How It Works
Prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.

*We use a totally different approach. We apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.*

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/model2.png" width="90%" height="90%">


# YOLOv3








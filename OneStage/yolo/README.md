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
# YOLOv1
*You Only Look Once:Unified, Real-Time Object Detection - [[Paper]](https://arxiv.org/pdf/1506.02640.pdf)*
![The YOLO Detection System](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/yolov1.png)

Processing imageswith YOLO is simple and straightforward. 
* (1) resizesthe input image to 448×448.
* (2) runs a single convolutional network on the image.
* (3) thresholds the resulting detections bythe model’s confidence.

![The Model](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/yolov1_1.png)

It divides the image into an S × S grid and for each grid cell predicts B bounding boxes, confidence for those boxes,
and C class probabilities.These predictions are encoded as an __S × S × (B ∗ 5 + C)__ tensor.
For evaluating YOLO on P ASCAL VOC, They use S = 7, B = 2. P ASCAL VOC has 20 labelled classes so C = 20.


__b :Confidence as Pr(Object) ∗ IOU (truth | pred)__
    
    If no object exists in that cell, the confidence scores should be zero. (Pr(Object) = 0)
    intersection over union (IOU) between the predicted box and the ground truth.
    
__c :Class probabilities, Pr(Class i |Object)__

__Pr(Class i |Object) ∗ Pr(Object) ∗ IOU(truth | pred) = Pr(Class i ) ∗ IOU(truth | pred)__

## Network Design
<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/yolov1network.png" width="90%" height="90%">

*Network has 24 convolutional layers followed by 2 fully connected layers. Alternating 1×1 convolutional layers reduce the features space from preceding layers.*

__The final prediction is a 7 × 7 × 30 tensor.__



***
# YOLOv2
### How It Works
Prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.

*We use a totally different approach. We apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.*

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/model2.png" width="90%" height="90%">


# YOLOv3








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

The YOLO (Redmon et al., 2016) model is the very first attempt at building a fast real-time object detector. Because YOLO does not undergo the region proposal step and only predicts over a limited number of bounding boxes, it is able to do inference super fast.
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
# YOLOv2 / YOLO9000

*YOLO9000: Better, Faster, Stronger - [[Paper]](https://arxiv.org/abs/1612.08242)*

YOLOv2 (Redmon & Farhadi, 2017) is an enhanced version of YOLO. YOLO9000 is built on top of YOLOv2 but trained with joint dataset combining the COCO detection dataset and the top 9000 classes from ImageNet.

### How It Works
Prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.

*We use a totally different approach. We apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.*

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/model2.png" width="50%" height="50%">

## YOLOv2 Improvement

A variety of modifications are applied to make YOLO prediction more accurate and faster, including:

* 1. BatchNorm helps: Add batch norm on all the convolutional layers, leading to significant improvement over convergence.

* 2. Image resolution matters: Fine-tuning the base model with high resolution images improves the detection performance.

* 3. Convolutional anchor box detection: Rather than predicts the bounding box position with fully-connected layers over the whole feature map, YOLOv2 uses convolutional layers to predict locations of anchor boxes, like in faster R-CNN. The prediction of spatial locations and class probabilities are decoupled. Overall, the change leads to a slight decrease in mAP, but an increase in recall.

* 4. K-mean clustering of box dimensions: Different from faster R-CNN that uses hand-picked sizes of anchor boxes, YOLOv2 runs k-mean clustering on the training data to find good priors on anchor box dimensions. The distance metric is designed to rely on IoU scores: __dist(x,ci)=1−IoU(x,ci),i=1,…,k__

*where x is a ground truth box candidate and ci is one of the centroids. The best number of centroids (anchor boxes) k
can be chosen by the elbow method.*

The anchor boxes generated by clustering provide better average IoU conditioned on a fixed number of boxes.

* 5. Direct location prediction: YOLOv2 formulates the bounding box prediction in a way that it would not diverge from the center location too much. If the box location prediction can place the box in any part of the image, like in regional proposal network, the model training could become unstable.

Given the anchor box of size (pw,ph) at the grid cell with its top left corner at (cx,cy), the model predicts the offset and the scale, (tx,ty,tw,th) and the corresponding predicted bounding box b has center (bx,by) and size (bw,bh). The confidence score is the sigmoid (σ) of another output to.

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/yolov2.png" width="60%" height="60%">

*YOLOv2 bounding box location prediction. (Image source: original paper)*

* 6. Add fine-grained features: YOLOv2 adds a passthrough layer to bring fine-grained features from an earlier layer to the last output layer. The mechanism of this passthrough layer is similar to identity mappings in ResNet to extract higher-dimensional features from previous layers. This leads to 1% performance increase.

* 7. Multi-scale training: In order to train the model to be robust to input images of different sizes, a new size of input dimension is randomly sampled every 10 batches. Since conv layers of YOLOv2 downsample the input dimension by a factor of 32, the newly sampled size is a multiple of 32.

* 8. Light-weighted base model: To make prediction even faster, YOLOv2 adopts a light-weighted base model, DarkNet-19, which has 19 conv layers and 5 max-pooling layers. The key point is to insert avg poolings and 1x1 conv filters between 3x3 conv layers.

# YOLOv3
*YOLOv3: An Incremental Improvement - [[Paper]](https://arxiv.org/abs/1804.02767)*

YOLOv3 is created by applying a bunch of design tricks on YOLOv2. The changes are inspired by recent advances in the object detection world.

Here are a list of changes:

* 1. Logistic regression for confidence scores: YOLOv3 predicts an confidence score for each bounding box using logistic regression, while YOLO and YOLOv2 uses sum of squared errors for classification terms (see the loss function above). Linear regression of offset prediction leads to a decrease in mAP.

* 2. No more softmax for class prediction: When predicting class confidence, YOLOv3 uses multiple independent logistic classifier for each class rather than one softmax layer. This is very helpful especially considering that one image might have multiple labels and not all the labels are guaranteed to be mutually exclusive.

* 3. Darknet + ResNet as the base model: The new Darknet-53 still relies on successive 3x3 and 1x1 conv layers, just like the original dark net architecture, but has residual blocks added.

* 4. Multi-scale prediction: Inspired by image pyramid, YOLOv3 adds several conv layers after the base feature extractor model and makes prediction at three different scales among these conv layers. In this way, it has to deal with many more bounding box candidates of various sizes overall.

* 5. Skip-layer concatenation: YOLOv3 also adds cross-layer connections between two prediction layers (except for the output layer) and earlier finer-grained feature maps. The model first up-samples the coarse feature maps and then merges it with the previous features by concatenation. The combination with finer-grained information makes it better at detecting small objects.
Interestingly, focal loss does not help YOLOv3, potentially it might be due to the usage of λnoobj and λcoordthey increase the loss from bounding box location predictions and decrease the loss from confidence predictions for background boxes.Overall YOLOv3 performs better and faster than SSD, and worse than RetinaNet but 3.8x faster.


# Reference
[1] Joseph Redmon, et al. “You only look once: Unified, real-time object detection.” CVPR 2016.

[2] Joseph Redmon and Ali Farhadi. “YOLO9000: Better, Faster, Stronger.” CVPR 2017.

[3] Joseph Redmon, Ali Farhadi. “YOLOv3: An incremental improvement.”.

[4] Lilian Weng. Object Detection Part 4: Fast Detection Models Dec 27, 2018

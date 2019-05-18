# SDD: Single Shot MultiBox Detector
*The Single Shot Detector (SSD; Liu et al, 2016) is one of the first attempts at using convolutional neural network’s pyramidal feature hierarchy for efficient detection of objects of various sizes.*

## Image Pyramid

SSD uses the VGG-16 model pre-trained on ImageNet as its base model for extracting useful image features. On top of VGG16, SSD adds several conv feature layers of decreasing sizes. They can be seen as a pyramid representation of images at different scales. Intuitively large fine-grained feature maps at earlier levels are good at capturing small objects and small coarse-grained feature maps can detect large objects well. In SSD, the detection happens in every pyramidal layer, targeting at objects of various sizes.

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/ssd/ssd_img/SSD-architecture.png" width="100%" height="100%">

## Workflow
Unlike YOLO, SSD does not split the image into grids of arbitrary size but predicts offset of predefined anchor boxes (this is called “default boxes” in the paper) for every location of the feature map. Each box has a fixed size and position relative to its corresponding cell. All the anchor boxes tile the whole feature map in a convolutional manner.

Feature maps at different levels have different receptive field sizes. The anchor boxes on different levels are rescaled so that one feature map is only responsible for objects at one particular scale. For example, in Fig. 5 the dog can only be detected in the 4x4 feature map (higher level) while the cat is just captured by the 8x8 feature map (lower level).

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/ssd/ssd_img/SSD-framework.png" width="80%" height="80%">

*The SSD framework. (a) The training data contains images and ground truth boxes for every object. (b) In a fine-grained feature maps (8 x 8), the anchor boxes of different aspect ratios correspond to smaller area of the raw input. (c) In a coarse-grained feature map (4 x 4), the anchor boxes cover larger area of the raw input. (Image source: original paper)*

The width, height and the center location of an anchor box are all normalized to be (0, 1). At a location (i,j) of the ℓ-th feature layer of size m×n, i=1,…,n,j=1,…,m, we have a unique linear scale proportional to the layer level and 5 different box aspect ratios (width-to-height ratios), in addition to a special scale (why we need this? the paper didn’t explain. maybe just a heuristic trick) when the aspect ratio is 1. This gives us 6 anchor boxes in total per feature cell.

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/ssd/ssd_img/proof.png" width="50%" height="50%">

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/ssd/ssd_img/SSD-box-scales.png" width="50%" height="50%">

*An example of how the anchor box size is scaled up with the layer index ℓ for L=6,smin=0.2,smax=0.9. Only the boxes of aspect ratio r=1 are illustrated.*

At every location, the model outputs 4 offsets and c class probabilities by applying a 3×3×p conv filter (where p is the number of channels in the feature map) for every one of k anchor boxes. Therefore, given a feature map of size m×n, we need kmn(c+4) prediction filters.

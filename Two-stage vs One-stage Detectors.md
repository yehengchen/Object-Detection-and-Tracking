# Two-stage vs One-stage Detectors
### Comparison Two-stage and One-stage - [[YouTube]](https://www.youtube.com/watch?v=V4P_ptn2FF4)

### Two-stage Detectors
*找出物体（Region Proposals） -> 识别物体（Object Recognition）*

-_Models in the R-CNN family are all region-based._ - [[R-CNN]]()
* First, the model proposes a set of regions of interests by select search or regional proposal network. The proposed regions are sparse as the potential bounding box candidates can be infinite. 
* Then a classifier only processes the region candidates.

        
![](https://github.com/yehengchen/ObjectDetection/blob/master/img/two_stage.png)

*The other different approach skips the region proposal stage and runs detection directly over a dense sampling of possible locations. This is how a one-stage object detection algorithm works. This is faster and simpler, but might potentially drag down the performance a bit.*

### One-stage Detectors
*找出物体同时识别物体 - Detecting objects in images using a single deep neural network*

-YOLO (You only look once): YOLOv1, YOLOv2, YOLOv3, Tiny YOLO - [[YOLO]](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo.md)

-Single Shot Detector (SSD) - [[SSD]]()

* Single convolutional network predicts the bounding boxes and the class probabilities for these boxes.

![](https://github.com/yehengchen/ObjectDetection/blob/master/img/one_stage.png)

## Two-stage vs One-stage Detectors
![](https://github.com/yehengchen/ObjectDetection/blob/master/img/yolo_vs_rcnn.png)

## Result on COCO
*For the last couple years, many results are exclusively measured with the COCO object detection dataset. COCO dataset is _harder_ for object detection and usually detectors achieve much lower mAP. Here are the comparison for some key detectors.*
![](https://github.com/yehengchen/ObjectDetection/blob/master/img/COCO%20object%20detection%20dataset.jpeg)

## Result on PASCAL VOC
*For the result presented below, the model is trained with both PASCAL VOC 2007 and 2012 data. The mAP is measured with the PASCAL VOC 2012 testing set. For SSD, the chart shows results for 300 × 300 and 512 × 512 input images. For YOLO, it has results for 288 × 288, 416 ×461 and 544 × 544 images. Higher resolution images for the same model have better mAP but slower to process.*

![](https://github.com/yehengchen/ObjectDetection/blob/master/img/PASCAL%20VOC%202007%20and%202012%20data.png)

*Input image resolutions and feature extractors impact speed. Below is the highest and lowest FPS reported by the corresponding papers. Yet, the result below can be highly biased in particular they are measured at different mAP.*

![](https://github.com/yehengchen/ObjectDetection/blob/master/img/PASCAL%20VOC%202007%20and%202012%20data%20FPS.png)

__Comparison COCO and Pascal VOC dataset__ -> [[Click Here]](https://github.com/yehengchen/ObjectDetection/blob/master/COCO%20and%20Pascal%20VOC.md)

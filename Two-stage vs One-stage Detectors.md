# Two-stage vs One-stage Detectors
### Comparison Two-stage and One-stage - [[YouTube]](https://www.youtube.com/watch?v=V4P_ptn2FF4)

### Two-stage Detectors
*找出物体（Region Proposals） -> 识别物体（Object Recognition）*

-_Models in the R-CNN family are all region-based._ - [[R-CNN]]()
* (1) First, the model proposes a set of regions of interests by select search or regional proposal network. The proposed regions are sparse as the potential bounding box candidates can be infinite. 
* (2) Then a classifier only processes the region candidates.

        
![](https://github.com/yehengchen/ObjectDetection/blob/master/img/two_stage.png)

The other different approach skips the region proposal stage and runs detection directly over a dense sampling of possible locations. This is how a one-stage object detection algorithm works. This is faster and simpler, but might potentially drag down the performance a bit.

### One-stage Detectors
*找出物体同时识别物体 - Detecting objects in images using a single deep neural network*

-YOLO (You only look once): YOLOv1, YOLOv2, YOLOv3, Tiny YOLO - [[YOLO]](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo.md)

-Single Shot Detector (SSD) - [[SSD]]()
![](https://github.com/yehengchen/ObjectDetection/blob/master/img/one_stage.png)

## Two-stage vs One-stage Detectors
![](https://github.com/yehengchen/ObjectDetection/blob/master/img/yolo_vs_rcnn.png)

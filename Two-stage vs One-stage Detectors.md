# Two-stage vs One-stage Detectors
Two-stage vs One-stage Detectors video ![[Youtube]](https://www.youtube.com/watch?v=V4P_ptn2FF4)
### Two-stage Detectors
-_Models in the R-CNN family are all region-based._
* (1) First, the model proposes a set of regions of interests by select search or regional proposal network. The proposed regions are sparse as the potential bounding box candidates can be infinite. 
* (2) Then a classifier only processes the region candidates.

The other different approach skips the region proposal stage and runs detection directly over a dense sampling of possible locations. This is how a one-stage object detection algorithm works. This is faster and simpler, but might potentially drag down the performance a bit.

### One-stage Detectors
-YOLO (You only look once): YOLOv1, YOLOv2, YOLOv3, Tiny YOLO

-Single Shot Detector (SSD)

![](https://github.com/yehengchen/ObjectDetection/blob/master/img/yolo_vs_rcnn.png)

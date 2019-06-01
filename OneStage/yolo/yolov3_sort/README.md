# YOLOv3 Sort Person Counter

The purpose of this project is to detect and track vehicles on a video stream and count those going through a defined line. 

![highway.gif](highway.gif)

It uses:

* [YOLOv3](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3) to detect objects on each of the video frames.

* [SORT](https://github.com/abewley/sort) to track those objects over different frames.

Once the objects are detected and tracked over different frames a simple mathematical calculation is applied to count the intersections between the vehicles previous and current frame positions with a defined line.

The code on this prototype uses the code structure developed by Adrian Rosebrock for his article [YOLO object detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv).

## Quick Start

1. Download the code to your computer.
2. [Download yolov3.weights](https://www.dropbox.com/s/99mm7olr1ohtjbq/yolov3.weights?dl=0) and place it in `/yolo-coco`.
3. Run:
```
$ python3 main.py --input input/test.mp4 --output output/test.avi --yolo yolo-obj
```

## Citation

### YOLO :

    @article{redmon2016yolo9000,
      title={YOLO9000: Better, Faster, Stronger},
      author={Redmon, Joseph and Farhadi, Ali},
      journal={arXiv preprint arXiv:1612.08242},
      year={2016}
    }

### SORT :

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }

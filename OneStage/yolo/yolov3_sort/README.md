# YOLOv3 + SORT - Person Counter

*This project is to detect and track person on a video stream and count those going through a defined line.*

![sort.gif](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/sort_1.gif)

## Requirement

* Python 3.5
* OpenCV
* Numpy

It uses:

* [YOLOv3](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3) to detect objects on each of the video frames.

* [SORT](https://github.com/abewley/sort) to track those objects over different frames.

Once the objects are detected and tracked over different frames a simple mathematical calculation is applied to count the intersections between the vehicles previous and current frame positions with a defined line.


## Quick Start

1. Download the code to your computer.
     
2. Download [[yolov3_person.weights]](https://yun.baidu.com/disk/home?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#/all?vmode=list&path=%2FGithub%2Fyolov3) and place it in `yolov3_sort/yolo-obj/`

3. [yolov3_sort/main.py] Change the Path to __labelsPath / weightsPath / configPath__.

4. Run the yolov3 counter:
```
$ python3 main.py --input input/test.mp4 --output output/test.avi --yolo yolo-obj
```

## Reference

### YOLOv3 :

    @article{yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Redmon, Joseph and Farhadi, Ali},
    journal = {arXiv},
    year={2018}
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
#### Github@[guillelopez]


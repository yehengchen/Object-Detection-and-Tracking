# YOLOv3 + Deep_SORT - Object Tracker & Counter

![deep_sort.gif](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolo_img/output.gif)

__Demo - [[BiliBili]](https://www.bilibili.com/video/av54470007/?p=2)__
## Requirement

* OpenCV
* NumPy
* sklean
* Pillow
* tensorflow-gpu 1.10.0 
***

It uses:

* [YOLOv3](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3) to detect objects on each of the video frames. - 用自己的数据训练Yolov3模型

* [Deep_SORT](https://github.com/nwojke/deep_sort) to track those objects over different frames.

*This repository contains code for Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT). We extend the original SORT algorithm to integrate appearance information based on a deep appearance descriptor. See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information.*


## Quick Start

1. Download the code to your computer.
     
2. Download [[yolov3_person.weights]](https://yun.baidu.com/disk/home?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#/all?vmode=list&path=%2FGithub%2Fyolov3) and place it in `yolov3_sort/yolo-obj/`

3. Convert the Darknet YOLO model to a Keras model:
```
$ python convert.py yolov3_1.cfg yolov3_person.weights model_data/yolo.h5
``` 

4. Run the YOLO_DEEP_SORT:

```
$ python demo.py
```

## Training on Market1501
[cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning) for training a metric feature representation to be used with the deep_sort tracker.

## Citing

### YOLOv3 :

    @article{yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Redmon, Joseph and Farhadi, Ali},
    journal = {arXiv},
    year={2018}
    }

### Deep_SORT :

    @inproceedings{Wojke2017simple,
    title={Simple Online and Realtime Tracking with a Deep Association Metric},
    author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
    booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
    year={2017},
    pages={3645--3649},
    organization={IEEE},
    doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
    title={Deep Cosine Metric Learning for Person Re-identification},
    author={Wojke, Nicolai and Bewley, Alex},
    booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
    year={2018},
    pages={748--756},
    organization={IEEE},
    doi={10.1109/WACV.2018.00087}
    }
    
## Reference
#### Github:deep_sort@[Nicolai Wojke nwojke](https://github.com/nwojke/deep_sort)
#### Github:deep_sort_yolov3@[Qidian213 ](https://github.com/Qidian213/deep_sort_yolov3)




# YOLOv3 + Deep_SORT

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/img/output_49.gif" width="40%" height="40%"> <img src="https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/yolo_img/TownCentreXVID_output_ss.gif" width="40%" height="40%">
<img src="https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/yolo_img/output_person_315_1120_s.gif" width="40%" height="40%"> <img src="https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/img/output_car_143.gif" width="40%" height="40%">

__Object Tracking & Counting Demo - [[YouTube]](https://www.youtube.com/watch?v=ALw3OfrGWGo) [[BiliBili_V1]](https://www.bilibili.com/video/av55778717) [[BiliBili_V2]](https://www.bilibili.com/video/av59454144/?p=1)  [[Chinese Version]](https://blog.csdn.net/weixin_38107271/article/details/96741706)__
## Requirement
__Development Environment:[Deep-Learning-Environment-Setup](https://github.com/yehengchen/Ubuntu-16.04-Deep-Learning-Environment-Setup)__ 

* OpenCV
* NumPy
* sklean
* Pillow
* tensorflow-gpu 1.10.0 
***

It uses:

* __Detection__: [YOLOv3](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3) to detect objects on each of the video frames. - 用自己的数据训练YOLOv3模型

* __Tracking__: [Deep_SORT](https://github.com/nwojke/deep_sort) to track those objects over different frames.

*This repository contains code for Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT). We extend the original SORT algorithm to integrate appearance information based on a deep appearance descriptor. See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information.*

## Quick Start

__1. Download the code to your computer.__
    
    git clone https://github.com/yehengchen/Object-Detection-and-Tracking.git
    
__2. Download [[yolov3.weights]](https://pjreddie.com/media/files/yolov3.weights)__ and place it in `yolov3_sort/yolo-obj/`

*__Here you can download my trained [[yolo_cc_0612.h5]](https://drive.google.com/open?id=1MJBmDxMgPDTno-5DRvnpVth10Rnu-DWO) weights for detection person/car/bicycle,etc.__*

__3. Convert the Darknet YOLO model to a Keras model:__
```
$ python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
``` 
__4. Run the YOLO_DEEP_SORT:__

```
$ python main.py -c [CLASS NAME] -i [INPUT VIDEO PATH]

$ python main.py -c person -i ./test_video/testvideo.avi
```

__5. Can change [deep_sort_yolov3/yolo.py] `__Line 100__` to your tracking target__

*DeepSORT pre-trained weights using people-ReID datasets only for person, other targets is not good*
```
    if predicted_class != args["class"]:
               continue
    
    if predicted_class != 'person' and predicted_class != 'car':
               continue
```

## Train on Market1501 & MARS
*People Re-identification model*

[cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning) for training a metric feature representation to be used with the deep_sort tracker.

## Citation

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




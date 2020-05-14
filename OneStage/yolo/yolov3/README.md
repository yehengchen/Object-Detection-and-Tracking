# How to train YOLOv3 model
__[[how-to-train-to-detect-your-custom-objects]](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)__
### Requirement
* Python 3.5
* OpenCV if you want a wider variety of supported image types.
* CUDA if you want GPU computation.


### Download the source code
    git clone https://github.com/pjreddie/darknet
    cd darknet
    
    vim Makefile
    ...
    GPU=1
    CUDNN=1
    NVCC=/usr/local/cuda-9.0/bin/nvcc
    OPENCV=1
    
    make
### Download pre-training weights
    wget https://pjreddie.com/media/files/yolov3.weights
### Pre-trainnig model testing
    ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
    
***
# Train a model on own images - 训练自己的数据
## 1. Image labeling - 标记数据集
#### LabelImg is a graphical image annotation tool - [labelImg](https://github.com/tzutalin/labelImg)
__Ubuntu Linux__
Python 3 + Qt5   
    
    git clone https://github.com/tzutalin/labelImg.git
    sudo apt-get install pyqt5-dev-tools
    sudo pip3 install -r requirements/requirements-linux-python3.txt
    make qt5py3
    cd labelImg
    
    python3 labelImg.py
    python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]


* __[JPEGImages](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/JPEGImages) [Store all [.jpg] img in this folder]__
* __[Annotations](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/Annotations) [Store all labeled [.xml] file in this folder]__ 
            
      [labeled .xml file2 .txt file]
      python3 voc_label.py
      
* __[labels](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/labels) [Transfer all labeled 2 [.txt] file in this folder and put all [.txt] file to JPEGImages folder]__

. <br>
__├── [JPEGImages](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/JPEGImages) <br>__
│   ├── object_00001.jpg <br>
│   └── object_00002.jpg <br>
│   ... <br>
__├── [Annotations](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/Annotations) <br>__
│   ├── object_00001.xml <br>
│   └── object_00002.xml <br>
│   ... <br>
__├── [labels](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/labels) <br>__
│   ├── object_00001.txt <br>
│   └── object_00002.txt <br>
│   ... <br>
__├── [backup](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/backup) <br>__
│   ├── yolov3-voc-object.backup <br>
│   └── yolov3-voc-object_20000.weights <br>
│   ... <br>
__├── [cfg](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/cfg) <br>__
│   ├── yolo3_object.data <br>
│   └── yolov3-voc-object.cfg  <br>
└── test <br>
***


## 2. Make .txt file - 制作 yolo 需要的文档
__Run *[img2train.py](https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/yolov3/img2train.py)*__ ： 将图像分为训练和验证集，保存为train.txt和val.txt
```
input: 
    python3 img2train.py /home/andy/Data/img
output: 
    ./train.txt
    ./val.txt
```
* [train.txt](https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/yolov3/train.txt):写入用于训练图片的名字，每行一个名字（不带后缀.jpg） - Store all train_img name without .jpg

* [val.txt](https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/yolov3/val.txt):写入用于验证图片的名字，每行一个名字（不带后缀.jpg） - Store all val_img name without .jpg


__Run *[voc_label.py](https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/yolov3/voc_label.py)* can get below file__

* [object_train.txt](https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/yolov3/object_train.txt):写入用于训练图片的绝对路径，每行一个路径。

* [object_val.txt](https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/yolov3/object_val.txt):写入用于验证图片的绝对路径，每行一个路径。

## 3. Make .names .cgf and .data file 
* __.names [classes name]__
*data folder voc.names*

    person
    fire_extinguisher
    fireplug
    car
    bicycle
    motorcycle

* __.data__ 
*cfg folder voc.data*
     
      classes= 6  #类别数
      train = /home/cai/Desktop/yolo_dataset/objectdetection/object_train.txt #obj_train.txt路径
      valid = /home/cai/Desktop/yolo_dataset/objectdetection/object_val.txt  #obj_val.txt路径
      names = /home/cai/Desktop/yolo_dataset/objectdetection/yolo3_object.names #obj_voc.names路径
      backup = /home/cai/Desktop/yolo_dataset/objectdetection/backup/ #建一个backup文件夹用于存放weights结果
 
 * __.cgf__
 *cfg folder yolov3-voc.cfg - __[example.cfg](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolov3/cfg/example.cfg)__*
       
       [convolutional]
       ...
       filters = 3*(classes + 5) #修改filters数量
       [yolo]
       ...
       classes=5 #修改类别数
       [具体修改可见cfg文件]
       
## 4. Download pre-taining weights -下载预训练 weights
    wget https://pjreddie.com/media/files/darknet53.conv.74
## 5. Training
    ./darknet detector train [path to .data file] [path to .cfg file] [path to pre-taining weights-darknet53.conv.74]
    
    ./darknet detector train obj_detect/cfg/obj_voc.data obj_detect/cfg/yolov3-voc.cfg darknet53.conv.74
    
    [visualization]
    ./darknet detector train obj_detect/cfg/obj_voc.data obj_detect/cfg/yolov3-voc.cfg darknet53.conv.74 2>1 | tee visualization/train_yolov3.log

#### Log Visualization - 训练可视化
__在 extract_log.py 中修改 train_yolov3.log 路径__
    
    python3 extract_log.py
    python3 visualization_loss.py
    python3 visualization_iou.py


***
## When should-i stop training - 什么时候停止训练

1. During training, you will see varying indicators of error, and you should stop when no longer decreases 0.XXXXXXX avg:

       Region Avg IOU: 0.798363, Class: 0.893232, Obj: 0.700808, No Obj: 0.004567, Avg Recall: 1.000000, count: 8 Region Avg IOU: 0.800677, Class: 0.892181, Obj: 0.701590, No Obj: 0.004574, Avg Recall: 1.000000, count: 8
       
       9002: 0.211667, 0.060730 avg, 0.001000 rate, 3.868000 seconds, 576128 images Loaded: 0.000000 seconds

* 9002 - iteration number (number of batch)
* 0.060730 avg - average loss (error) - the lower, the better

For details __[when-should-i-stop-training](https://github.com/AlexeyAB/darknet#when-should-i-stop-training)__

#### Training log

    Avg IOU:当前迭代中，预测的box与标注的box的平均交并比，越大越好，期望数值为1；
    Class: 标注物体的分类准确率，越大越好，期望数值为1；
    obj: 越大越好，期望数值为1；
    No obj: 越小越好；
    .5R: 以IOU=0.5为阈值时候的recall; recall = 检出的正样本/实际的正样本
    0.75R: 以IOU=0.75为阈值时候的recall;
    count:正样本数目。 
    
    1: 1452.927612, 1452.927612 avg, 0.000000 rate, 1.877576 seconds, 32 images
    第几批次，总损失，平均损失，当前学习率，当前批次训练时间，目前为止参与训练的图片总数
    1： 指示当前训练的迭代次数
    1452.927612： 是总体的Loss(损失）
***

##  Train a model on VOC2007 / VOC2012 / COCO2017

### VOC2007 Data

* Download the [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (450MB tar file)

### VOC2012 Data

* Download the [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (2GB tar file)

### COCO2017 Data

* Download the [2017 Train images](http://images.cocodataset.org/zips/train2017.zip) [118K/18GB]
* Download the [2017 Val images](http://images.cocodataset.org/zips/val2017.zip) [5K/1GB]
* Download the [2017 Test images](http://images.cocodataset.org/zips/test2017.zip) [41K/6GB]
* COCO API/[PythonAPI](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI)

***

## TEST

### ImgTesting
    ./darknet detector test ./obj_detect/cfg/obj_voc.data ./obj_detect/cfg/yolov3-voc.cfg ./obj_detect/backup/yolov3-voc_30000.weights ./obj_detect/test/test_img.jpg
### VideoTesting
    ./darknet detector demo ../obj_detect/cfg/obj_voc.data ./obj_detect/cfg/yolov3-voc.cfg ./obj_detect/backup/yolov3-voc_30000.weights ./obj_detect/test/obj_test.mp4

![](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolov3/test_img/predictions.jpg)

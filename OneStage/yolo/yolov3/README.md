# YOLOv3
## Geting start
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
# Train a model on own images
## 1. Image labeling 
#### LabelImg is a graphical image annotation tool - [labelImg](https://github.com/tzutalin/labelImg)
__Ubuntu Linux__
Python 3 + Qt5   
    
    sudo apt-get install pyqt5-dev-tools
    sudo pip3 install -r requirements/requirements-linux-python3.txt
    make qt5py3
    python3 labelImg.py
    python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

* [JPEGImages](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/JPEGImages) [Put all img in this folder]
* [Annotations](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/Annotations) [Put all labeled .xml file in this folder]
* [labels](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/labels) [Put all labeled .txt file in this folder]

## 2. Make .txt file

* train.txt:存放用于训练的图片的名字，每行一个名字（不带后缀.jpg）。

* val.txt:存放用于验证的图片的名字，每行一个名字（不带后缀.jpg）。

__Run voc_label.py can get below file__

* obj_train.txt:存放用于训练的图片的绝对路径，每行一个路径。

* obj_val.txt:存放用于验证的图片的绝对路径，每行一个路径。

## 3. Make .names and .data file 
* __.names [classes name]__
*data folder voc.names*
* __.data__ 
*cfg folder voc.data*
     
      classes= 5  #类别数
      valid  = /home/cai/darknet/obj_detect/obj_val.txt  #boat_val.txt路径
      names = /home/cai/darknet/obj_detect/obj_voc.names #boat_voc.names路径
      backup = /home/cai/darknet/obj_detect/backup/ #建一个backup文件夹用于存放中间结果
 * __.cgf__
 *cfg folder yolov3-voc.cfg*
      
    [net]
    # Testing
    # batch=1
    # subdivisions=1    #训练时候把上面Testing的参数注释
    # Training
    batch=64
    subdivisions=32     #这个参数根据自己GPU的显存进行修改，显存不够就改大一些
    ...                 #因为训练时每批的数量 = batch/subdivisions
    ...
    ...
    learning_rate=0.001  #根据自己的需求还有训练速度学习率可以调整一下
    burn_in=1000
    max_batches = 30000  #根据自己的需求还有训练速度max_batches可以调整一下
    policy=steps
    steps=10000,20000    #跟着max_batches做相应调整
    ...
    ...
    ...
    [convolutional]
    size=1
    stride=1
    pad=1
    filters=30         #filters = 3*(classes + 5)
    activation=linear

    [yolo]
    mask = 0,1,2
    anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    classes=5          #修改类别数
    num=9
    jitter=.3
    ignore_thresh = .5
    truth_thresh = 1
    random=1           #显存小的话 =0

    #这个文件的最下面有3个YOLO层，这里我才放上来了一个，这三个地方的classes做相应修改
    #每个YOLO层的上一层的convolutional层的filters也要修改

## 4. Download pre-taining weights
    wget https://pjreddie.com/media/files/darknet53.conv.74
## 5. Training
    ./darknet detector train obj_detect/obj_voc.data obj_detect/yolov3-voc.cfg darknet53.conv.74 
## 6. Testing
### ImgTesting
    ./darknet detector test ./obj_detect/obj_voc.data ./obj_detect/yolov3-voc.cfg ./obj_detect/backup/yolov3-voc_30000.weights ./obj_detect/test_data/test_img.jpg
### VideoTesting
    ./darknet detector demo ./obj_detect/obj_voc.data ./obj_detect/yolov3-voc.cfg ./obj_detect/backup/yolov3-voc_30000.weights ./obj_detect/test_data/obj_test.mp4
    

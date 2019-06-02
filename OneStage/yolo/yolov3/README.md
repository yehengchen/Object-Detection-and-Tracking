# Train YOLOv3 model

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
# Train a model on own images
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


* __[JPEGImages](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/JPEGImages) [Put all [.jpg] img in this folder]__
* __[Annotations](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3/Annotations) [Put all labeled [.xml] file in this folder]__ 
            
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



## 2. Make .txt file

* train.txt:写入用于训练图片的名字，每行一个名字（不带后缀.jpg）。

* val.txt:写入用于验证图片的名字，每行一个名字（不带后缀.jpg）。


__Run voc_label.py can get below file__

* object_train.txt:写入用于训练图片的绝对路径，每行一个路径。

* object_val.txt:写入用于验证图片的绝对路径，每行一个路径。

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
    
    # 在 extract_log.py 中修改 train_yolov3.log 路径
    python3 extract_log.py
    python3 visualization_loss.py
    python3 visualization_iou.py

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

##  Train a model on VOC2007 or VOC2012

### VOC2007 Data

* Download the [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (450MB tar file)

### VOC2012 Data

* Download the [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (2GB tar file)
***

## TEST

### ImgTesting
    ./darknet detector test ./obj_detect/cfg/obj_voc.data ./obj_detect/cfg/yolov3-voc.cfg ./obj_detect/backup/yolov3-voc_30000.weights ./obj_detect/test/test_img.jpg
### VideoTesting
    ./darknet detector demo ../obj_detect/cfg/obj_voc.data ./obj_detect/cfg/yolov3-voc.cfg ./obj_detect/backup/yolov3-voc_30000.weights ./obj_detect/test/obj_test.mp4

![](https://github.com/yehengchen/ObjectDetection/blob/master/OneStage/yolo/yolov3/test_img/predictions.jpg)

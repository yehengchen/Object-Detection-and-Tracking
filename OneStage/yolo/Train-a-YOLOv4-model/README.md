# How to train YOLOv4 model on own dataset

## Environment
 * Ubuntu 16.04/18.04
 * CUDA 10.0
 * cuDNN 7.6.0
 * Python 3.6
 * OpenCV 4.2.0
 * tensorflow-gpu 1.13.0
 
### Requirements

    pip3 install -r requirements.txt
       
### Download the source code

    git clone https://github.com/AlexeyAB/darknet.git
    cd darknet

    vim Makefile

    GPU=1
    CUDNN=1 
    CUDNN_HALF=1 
    OPENCV=1 
    DEBUG=1 
    OPENMP=1 
    LIBSO=1 
    ZED_CAMERA=1 
    
    make


### Download pre-trained weights file

[Google drive] - [yolov4.conv.137](https://drive.google.com/file/d/1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp/view)

[Baidu drive] - [yolov4.conv.137](https://pan.baidu.com/s/1OvuN0CeS7RFj-bdM0d25AA)  __code: nppt__

    
### Image labeling

LabelImg is a graphical image annotation tool - [labelImg](https://github.com/tzutalin/labelImg)

Ubuntu Linux Python5 + Qt5
         
    git clone https://github.com/tzutalin/labelImg.git
    sudo apt-get install pyqt5-dev-tools
    sudo pip3 install -r requirements/requirements-linux-python3.txt
    make qt5py3
    cd labelImg

    python3 labelImg.py
    python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
    
    
 * JPEGImages -- Store all __[.jpg]__ imgages
 * Annotations -- Store all labeled __[.xml]__ file
 * labels -- Store all __[.txt]__ file (convert all __[.xml]__ file to __[.txt]__ file)
   
       python3 ./tools/voc_label.py (convert xml2txt and check your file paths)
       
###  Make img path [.txt] file

#### First you have to devide your dataset into train dataset and validation dataset.

    python3 ./tools/img2train.py [img path]
      
 * train.txt -- Store all train_img name without .jpg
 * val.txt -- Store all val_img name without .jpg

#### Run [voc_label.py](https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/Train-a-YOLOv4-model/tools/voc_label.py) can get below file

 * object_train.txt -- Store all train_img __absolute path__
 * object_val.txt -- Store all val_img __absolute path__

### Make [.names] [.data] and [.cfg] file
 
 * __.names__ file
 
       gedit train.names
         
       class1
       class2
       class3
       class4
       ...
         
    Put your class list in train.names, save and quit.
 
 * __.data__ file
          
       gedit obj.data
          
       classes= [number of objects]
       train = [object_train.txt absolute path]
       valid = [object_val.txt absolute path]
       names = [train.names absolute path]
       backup = backup/ #save weights files here
     
    Put your class number and path in obj.data, save and quit.

 * __.cfg__ file stored in darknet/cfg/yolov4-custom.cfg (copy yolov4-custom.cfg to your folder)
 
    * change line batch to batch=64
    * change line subdivisions to subdivisions=16 (According to the GPU configuration, it can be adjusted to 32 or 64.)
    * change line max_batches to (classes*2000 but not less than number of training images, and not less than 6000), f.e. max_batches=6000 if you train for 3 classes
    * change line steps to __80% and 90%__ of max_batches, f.e. steps=4800,5400
    * set network size width=416 height=416 or any value multiple of 32: https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L8-L9
change line classes=80 to your number of objects in each of 3 [yolo]-layers:

      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L610
      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L696
      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L783

    
   * change __[filters=255]__ to __filters=(classes + 5)x3__ in the __3 [convolutional]__ before each __[yolo] layer__, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers.

      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L603
      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L689
      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L776

## Training

<img src="https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/Train-a-YOLOv4-model/imgs/chart_yolov4-custom.png" width="60%" height="60%">
 
  * Training and visualization
 
        sudo ./darknet detector train [obj.data path] [yolov4-custom.cfg path]  yolov4.conv.137 -map
        
  * Train with multi-GPU

        sudo ./darknet detector train [obj.data path] [yolov4-custom.cfg path]  yolov4.conv.137 -gpus 0,1,2 -map

 ## Testing

<img src="https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/Train-a-YOLOv4-model/imgs/yolov4.png" width="60%" height="60%">

   * Image test
   
         ./darknet detector test [obj.data path] [yolov4-custom.cfg path] [weights file path] [image path]
       
   * VIdeo test
   
         ./darknet detector demo [obj.data path] [yolov4-custom.cfg path] [weights file path] [video path]
        
      
   * If you want to save video test results
        
         ./darknet detector demo [obj.data path] [yolov4-custom.cfg path] [weights file path] [video path] -out_filename [Custom naming]

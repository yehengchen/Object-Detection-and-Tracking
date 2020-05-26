# How to train YOLOv4 model on own dataset

## Environment
 * Ubuntu 18.04
 * CUDA 10.0
 * cuDNN 7.6.0
 * Python 3.6
 * OpenCV 4.2.0
      
       pip3 install -r requirements.txt
       
## Download the source code

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
    
## Download pre-trained weights file

[yolov4.conv.137](https://drive.google.com/file/d/1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp/view)
    
## Image labeling

LabelImg is a graphical image annotation tool - [labelImg](https://github.com/tzutalin/labelImg)

Ubuntu Linux Python5 + Qt5
         
     git clone https://github.com/tzutalin/labelImg.git
     sudo apt-get install pyqt5-dev-tools
     sudo pip3 install -r requirements/requirements-linux-python3.txt
     make qt5py3
     cd labelImg

     python3 labelImg.py
     python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
    
    
 * JPEGImages -- Store all [.jpg] imgages
 * Annotations -- Store all labeled [.xml] file
 * labels -- Transfer all labeled [.xml] file to [.txt] file
   
       python3 ./tools/voc_label.py (change and check your file paths)
       
##  Make path[.txt] file

### First you have to devide your dataset into train dataset and validation dataset.

       python3 ./tools/img2train.py [img path]
      
 * train.txt -- Store all train_img name without .jpg
 * val.txt -- Store all val_img name without .jpg

### Run voc_label.py can get below file

 * object_train.txt -- Store all train_img paths
 * object_val.txt -- Store all val_img paths

 ## Make [.names] [.data] and [.cfg] file
 
 * .names file
 
        vim train.names
         
         class1
         class2
         class3
         class4
         ...
         
    Put your class list in train.names, save and quit.
 
 * .data file
          
          vim obj.data
          
          classes= [number of objects]
          train = [obj_train.txt path]
          valid = [obj_val.txt path]
          names = [train.names path]
          backup = backup/ #save weights files here
     
    Put your class number and path in obj.data, save and quit.

 * .cfg file stored in darknet/cfg/yolov4-custom.cfg(copy yolov4-custom.cfg to your folder)
 
    * change line batch to batch=64
    * change line subdivisions to subdivisions=16 (According to the GPU configuration, it can be adjusted to 32 or 64.)
    * change line max_batches to (classes*2000 but not less than number of training images, and not less than 6000), f.e. max_batches=6000 if you train for 3 classes
    * change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
    * set network size width=416 height=416 or any value multiple of 32: https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L8-L9
change line classes=80 to your number of objects in each of 3 [yolo]-layers:

      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L610
      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L696
      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L783

    
   * change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers.

      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L603
      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L689
      - https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L776

  ## Training
  
  * Training and visualization
 
        sudo ./darknet detector train [obj.data path] [yolov4-custom.cfg path]  yolov4.conv.137 -map
        
  * Train with multi-GPU

        sudo ./darknet detector train [obj.data path] [yolov4-custom.cfg path]  yolov4.conv.137 -gpus 0,1,2 -map
 
 ## Test
 
   * Image test
   
         ./darknet detector test [obj.data path] [yolov4-custom.cfg path] [weights file path] [image path]
       
   * VIdeo test
   
          ./darknet detector demo [obj.data path] [yolov4-custom.cfg path] [weights file path] [video path]
        
      
   * If you want to save video test results
        
           ./darknet detector demo [obj.data path] [yolov4-custom.cfg path] [weights file path] [video path] -out_filename [Custom naming]

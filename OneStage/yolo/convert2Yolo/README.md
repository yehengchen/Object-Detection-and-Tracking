# Convert2Yolo

Object Detection annotation Convert to [Yolo Darknet](https://pjreddie.com/darknet/yolo/) Format

Support DataSet : 

1. COCO
2. VOC
3. UDACITY Object Detection
4. KITTI 2D Object Detection

​    

## Pre-Requiredment

```
pip3 install -r requirements.txt
```

​    

## Required Parameters

each dataset requried some parameters

see [example.py](https://github.com/ssaru/convert2Yolo/blob/master/example.py)

1. --datasets
   - like a COCO / VOC / UDACITY / KITTI

     ```bash
     --datasets COCO
     ```
2. --img_path
   - it directory path. not file path

     ```bash
     --img_path ./example/kitti/images/
     ```
3. --label
   - it directory path. not file path

     (some datasets give label `*.json` or `*.csv` . this case use file path)

     ```bash
     --label ./example/kitti/labels/
     ```

     OR

     ```bash
     --label ./example/kitti/labels/label.json
     
     or
     
     --label ./example/kitti/labels/label.csv
     ```

4. --convert_output_path
   - it directory path. not file path

     ```bash
     --convert_output_path ./
     ```
5. --img_type
   - like a `*.png`, `*.jpg`

     ```bash
     --img_type ".jpg"
     ```
6. --manipast_path
   - it need train yolo model in [darknet framework](https://pjreddie.com/darknet/)

     ```bash
     --manipast_path ./
     ```
7. --cla_list_file(`*.names`)
   - it is `*.names` file contain class name. refer [darknet `*.name` file](https://github.com/pjreddie/darknet/blob/master/data/voc.names)

     ```bash
     --cls_list_file voc.names
     ```

​    

### *.names file example
```
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```

​    

## Example

​    

### 1. example command

```bash
python3 example.py --datasets [COCO/VOC/KITTI/UDACITY] --img_path <image_path> --label <label path or annotation file> --convert_output_path <output path> --img_type [".jpg" / ".png"] --manipast_path <output manipast file path> --cls_list_file <*.names file path>

>>
ex) python3 example.py --datasets KITTI --img_path ./example/kitti/images/ --label ./example/kitti/labels/ --convert_output_path ./ --img_type ".jpg" --manipast_path ./ --cls_list_file names.txt
```

​    

### 2. VOC datasets

​    

#### description of dataset directory

suppose that VOC dataset location are `~/VOC` and VOC folder contains `VOCdevkit` folder

here are structure for `VOCdevkit`

​    

**`VOCdevkit`**

```bash
$ tree -L 2
.
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    └── SegmentationObject
```

we use only `Annotations` and `JPEGImages` folder

- Annotations : Object Detection label folder
- JPEGImages : Image data

​    

**Annotations**

```bash
$ tree -L 1
.
├── 2007_000027.xml
├── 2007_000032.xml
├── 2007_000033.xml
...
├── 2012_004319.xml
├── 2012_004326.xml
├── 2012_004328.xml
├── 2012_004329.xml
├── 2012_004330.xml
└── 2012_004331.xml
```

​      

**JPEGImages**

```bash
.
├── 2007_000027.jpg
├── 2007_000032.jpg
├── 2007_000033.jpg
...
├── 2012_004328.jpg
├── 2012_004329.jpg
├── 2012_004330.jpg
└── 2012_004331.jpg
```

​    

#### make `*.names` file

now make `*.names` file in `~/VOC/`

refer [darknet `voc.names` file](https://github.com/pjreddie/darknet/blob/master/data/voc.names)

```bash
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```

​     

#### VOC datasets convert to YOLO format

now execute example code. 

this example assign directory for saving  `YOLO` label `~/YOLO/` and assign `manipast_path` is `./`

​    

**make YOLO folder**

```bash
$ mkdir ~/YOLO
```

​    

**VOC convert to YOLO**

```bash
python3 example.py --datasets VOC --img_path ~/VOCdevkit/VOC2012/JPEGImages/ --label ~/VOCdevkit/VOC2012/Annotations/ --convert_output_path ~/YOLO/ --img_type ".jpg" --manipast_path ./ --cls_list_file ~/VOC/voc.names

>>
VOC Parsing:   |████████████████████████████████████████| 100.0% (17125/17125) Complete
YOLO Generating:|████████████████████████████████████████| 100.0% (17125/17125)Complete
YOLO Saving:   |████████████████████████████████████████| 100.0% (17125/17125) Complete
```

​    

#### Result    

now check result files (`~/YOLO/`, `./manifast.txt`)

​    

**`~/YOLO/`**

```bash
$ tree -L 1
>>
├── 2012_004326.txt
├── 2012_004328.txt
├── 2012_004329.txt
├── 2012_004330.txt
└── 2012_004331.txt
...
├── 2012_004326.txt
├── 2012_004328.txt
├── 2012_004329.txt
├── 2012_004330.txt
└── 2012_004331.txt
```

​    

**`2012_004331.txt`**

```bash
$ cat 2012_004331.txt

>>
14 0.31 0.34 0.212 0.547
```

​    

**`./manifast.txt`**

```bash
$ cat ./manifast.txt

>>
~/VOC/VOCdevkit/VOC2012/JPEGImages/2010_000420.jpg
~/VOC/VOCdevkit/VOC2012/JPEGImages/2010_003674.jpg
~/VOC/VOCdevkit/VOC2012/JPEGImages/2012_002128.jpg
...
~/VOC/VOCdevkit/VOC2012/JPEGImages/2009_000104.jpg
~/VOC/VOCdevkit/VOC2012/JPEGImages/2012_000212.jpg
```



​    

### 3. COCO datasets

​    

#### description of dataset directory

suppose that COCO dataset location are `~/COCO` and COCO folder contains `annotations`, `val2017` folder

here are each structure for annotations and val2017

​    

**annotations**

```bash
$ cd ~/COCO/annotations/
$ tree -L 1
.
└── instances_val2017.json
```

​    

**val2017**

```bash
.
├── 000000000139.jpg
├── 000000000285.jpg
├── 000000000632.jpg
├── 000000000724.jpg
...
├── 000000581357.jpg
├── 000000581482.jpg
├── 000000581615.jpg
└── 000000581781.jpg

```

​    

#### make `*.names` file

now make `*.names` file in `~/COCO/`

refer [darknet `coco.names` file](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

```bash
person
bicycle
car
motorbike
aeroplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
sofa
pottedplant
bed
diningtable
toilet
tvmonitor
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
motorcycle
potted plant
dining table
tv
couch
airplane
```

​    

#### COCO datasets convert to YOLO format

now execute example code. 

this example assign directory for saving  `YOLO` label `~/YOLO/` and assign `manipast_path` is `./`

​    

**make YOLO folder**

```bash
$ mkdir ~/YOLO
```

​        

**COCO convert to YOLO**

```bash
python3 example.py --datasets COCO --img_path ~/COCO/val2017/ --label ~/COCO/annotations/instances_val2017.json --convert_output_path ~/YOLO/ --img_type ".jpg" --manipast_path ./ --cls_list_file ~/COCO/coco.names

>>
COCO Parsing:  |████████████████████████████████████████| 100.0% (36781/36781) Complete
YOLO Generating:|████████████████████████████████████████| 100.0% (4952/4952)  Complete
YOLO Saving:   |████████████████████████████████████████| 100.0% (4952/4952)  Complete
```

​        

#### Result    

now check result files (`~/YOLO/`, `./manifast.txt`)

**`~/YOLO/`**

```bash
.
├── 000000000139.txt
├── 000000000285.txt
├── 000000000632.txt
├── 000000000724.txt
...
├── 000000581206.txt
├── 000000581317.txt
├── 000000581357.txt
├── 000000581482.txt
├── 000000581615.txt
└── 000000581781.txt
```

​    

**`000000581781.txt`**

```bash
46 0.446 0.557 0.465 0.209
46 0.517 0.851 0.363 0.128
46 0.939 0.05 0.122 0.071
46 0.786 0.027 0.11 0.054
46 0.171 0.247 0.19 0.139
46 0.865 0.773 0.27 0.372
46 0.111 0.552 0.215 0.333
46 0.51 0.744 0.376 0.207
46 0.811 0.377 0.25 0.36
46 0.955 0.388 0.09 0.181
46 0.195 0.333 0.153 0.224
46 0.036 0.183 0.065 0.357
46 0.496 0.45 0.389 0.132
46 0.499 0.52 0.998 0.956
```

​        

**`./manifast.txt`**

```bash
~/COCO/val2017/000000289343.jpg
~/COCO/val2017/000000061471.jpg
~/COCO/val2017/000000472375.jpg
~/COCO/val2017/000000520301.jpg
~/COCO/val2017/000000579321.jpg
~/COCO/val2017/000000494869.jpg
...
~/COCO/val2017/000000097585.jpg
~/COCO/val2017/000000429530.jpg
~/COCO/val2017/000000031749.jpg
~/COCO/val2017/000000284282.jpg
```

​    

### TODO

- [x] Support VOC Pascal Format
- [x] Support Udacity Format
- [x] Support COCO Format
- [x] Support KITTI Format
- [x] Write README
- [x] Code Refactoring
- [ ] Add example coco about each datasets

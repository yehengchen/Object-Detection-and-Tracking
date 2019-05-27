import os
names = os.listdir('/home/cai/Desktop/object_detection/datasets/coco/result/images/')  #图片路径
i=0
train_val = open('/home/cai/Desktop/yolo_dataset/data/train.txt','w') #txt文件路径
for name in names: 
    index = name.rfind('.')
    name = name[:index]
    train_val.write(name+'\n')
    i=i+1
print(i)

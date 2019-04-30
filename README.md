# ObjectDetectionNote

## mAP
*mAP (mean average precision) is the average of AP. In some context, we compute the AP for each class and average them. But in some context, they mean the same thing.*

### AP
AP (Average precision) is a popular metric in measuring the accuracy of object detectors like Faster R-CNN, SSD, etc. Average precision computes the average precision value for recall value over 0 to 1. It sounds complicated but actually pretty simple as we illustrate it with an example. But before that, we will do a quick recap on precision, recall, and IoU first.

### Precision & recall
<img src="https://github.com/yehengchen/ObjectDetection/blob/master/img/fig1%20.png" width="40%" height="40%">


* Precision measures how accurate is your predictions. i.e. the percentage of your predictions are correct.
  
      Precision: TP / (TP + FP)
      (TP + FP) = Total Positive Result

* Recall measures how good you find all the positives. For example, we can find 80% of the possible positive cases in our top K predictions.
      
      Recall: TP / (TP + FN)
      (TP + FN) = Total Case
      
1. TP / True Positive: case was positive and predicted positive (IoU>0.5)
2. TN / True Negative: case was negative and predicted negative
3. FP / False Positive: case was negative but predicted positive(IoU<=0.5)
4. FN / False Negative: case was positive but predicted negative(没有检测到的GT的数量)
      
*Precision是确定分类器中断言为正样本的部分其实际中属于正样本的比例，精度越高则假的正例就越低，Recall则是被分类器正确预测的正样本的比例。
两者是一对矛盾的度量，其可以合并成令一个度量F1.*

![](https://github.com/yehengchen/ObjectDetection/blob/master/img/F1.png)

***

### IoU (Intersection over union)
IoU measures the overlap between 2 boundaries. We use that to measure how much our predicted boundary overlaps with the ground truth (the real object boundary). In some datasets, we predefine an IoU threshold __(say 0.5)__ in classifying whether the prediction is a true positive or a false positive.

![](https://github.com/yehengchen/ObjectDetection/blob/master/img/fig2.png)

### Interpolated AP
PASCAL VOC is a popular dataset for object detection. For the PASCAL VOC challenge, a prediction is positive if IoU ≥ 0.5. Also, if multiple detections of the same object are detected, it counts the first one as a positive while the rest as negatives.

In Pascal VOC2008, an average for the 11-point interpolated AP is calculated.

![](https://github.com/yehengchen/ObjectDetection/blob/master/img/fig1-2.jpeg)

First, we divide the recall value from 0 to 1.0 into 11 points — 0, 0.1, 0.2, …, 0.9 and 1.0. Next, we compute the average of maximum precision value for these 11 recall values.

![](https://github.com/yehengchen/ObjectDetection/blob/master/img/fig1-1.jpeg)

In our example, AP = (5 × 1.0 + 4 × 0.57 + 2 × 0.5)/11

Here are the more precise mathematical definitions.

![](https://github.com/yehengchen/ObjectDetection/blob/master/img/ap.png)

When APᵣ turns extremely small, we can assume the remaining terms to be zero. i.e. we don’t necessarily make predictions until the recall reaches 100%. 

***
### COCO mAP - [[cocodataset]](http://cocodataset.org/#detection-eval)
__COCO vs VOC - [[Link]](https://github.com/yehengchen/ObjectDetection/blob/master/COCO%20and%20Pascal%20VOC.md)__
Latest research papers tend to give results for the COCO dataset only. In COCO mAP, a 101-point interpolated AP definition is used in the calculation. For COCO, AP is the average over multiple IoU (the minimum IoU to consider a positive match). AP@[.5:.95] corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05. For the COCO competition, AP is the average over 10 IoU levels on 80 categories (AP@[.50:.05:.95]: start from 0.5 to 0.95 with a step size of 0.05). The following are some other metrics collected for the COCO dataset.

<img src="https://github.com/yehengchen/ObjectDetection/blob/master/img/fig3.png" width="80%" height="80%">
<img src="https://github.com/yehengchen/ObjectDetection/blob/master/img/fig4.png" width="80%" height="80%">

In the figure above, AP@.75 means the AP with IoU=0.75.

mAP (mean average precision) is the average of AP. In some context, we compute the AP for each class and average them. But in some context, they mean the same thing. For example, under the COCO context, there is no difference between AP and mAP. Here is the direct quote from COCO:

*AP is averaged over all categories. Traditionally, this is called “mean average precision” (mAP). We make no distinction between AP and mAP (and likewise AR and mAR) and assume the difference is clear from context.*
    
    

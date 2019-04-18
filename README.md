# ObjectDetectionNote

## mAP
*mAP (mean average precision) is the average of AP. In some context, we compute the AP for each class and average them. But in some context, they mean the same thing.*

### AP
AP (Average precision) is a popular metric in measuring the accuracy of object detectors like Faster R-CNN, SSD, etc. Average precision computes the average precision value for recall value over 0 to 1. It sounds complicated but actually pretty simple as we illustrate it with an example. But before that, we will do a quick recap on precision, recall, and IoU first.

### Precision & recall

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

![Fig]()

### IoU (Intersection over union)
IoU measures the overlap between 2 boundaries. We use that to measure how much our predicted boundary overlaps with the ground truth (the real object boundary). In some datasets, we predefine an IoU threshold __(say 0.5)__ in classifying whether the prediction is a true positive or a false positive.

![fig2]()


### COCO mAP -[[cocodataset]](http://cocodataset.org/#detection-eval)
Latest research papers tend to give results for the COCO dataset only. In COCO mAP, a 101-point interpolated AP definition is used in the calculation. For COCO, AP is the average over multiple IoU (the minimum IoU to consider a positive match). AP@[.5:.95] corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05. For the COCO competition, AP is the average over 10 IoU levels on 80 categories (AP@[.50:.05:.95]: start from 0.5 to 0.95 with a step size of 0.05). The following are some other metrics collected for the COCO dataset.

![fig3]()


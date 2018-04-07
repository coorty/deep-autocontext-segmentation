# deep-autocontext-segmentation
A combined of deep learned features and auto-context features used for object segmentation.

This repository is based on the Matlab implementation

这是下面论文的配套程序:

> **

本文方法将深度学习模型与Auto-context()特征相结合,采用多级级联的方式来实现目标的分割.

## Dataset
The model was primiaryly designed for segmenting myocardium tissues of cardiac MR images that comes from [The Cardiac Atlas Project](http://www.cardiacatlas.org/). To verify the generalization of the proposed model, we applied it to [Weizmann Horses Dataset](http://www.cardiacatlas.org/) segmentation without much modification. The Weizmann Horse Database consists of 328 side-view color images of horses that were also manually segmented, which seems like:

![](data/images/horse012.jpg) | ![](data/images/horse016.jpg) | ![](data/images/horse083.jpg)

![](data/labels/horse012.jpg) | ![](data/labels/horse016.jpg) | ![](data/labels/horse083.jpg)
  
## Model
The model combined deep learning and compentary features for pixel-level classification. 


## Results


**Note**: It is an object segmenting task, dont view it as an object detection task.
It is not object detection but pixel-wise classification. 


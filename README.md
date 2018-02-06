# Video Stream Object Detector
Xinhe Ren, Berkeley Autolab

## Overview
With the rise of live video streaming, a massive amount of data flows through the internet without getting collected. On the other hand, the recent advance in deep learning has shown the importance of big data. This repo features a pipeline for live stream video collection, as well as online object recognition in video streams.

## Dependencies
[OpenCV Python](https://opencv.org/)

## Object Recognition Architecture
### [Single Shot MultiBox Detector (SSD)](https://github.com/balancap/SSD-Tensorflow)
To use "SSD Detector", see [SSDDetector.py](https://github.com/renxinhe/Video-Stream-Object-Detector/blob/master/SSDDetector.py).

To obtain pre-trained weights, download [SSD-300 VGG-based](https://drive.google.com/file/d/0B0qPCUZ-3YwWUXh4UHJrd1RDM3c/view?usp=sharing) from the [SSD-Tensorflow repository](https://github.com/balancap/SSD-Tensorflow#evaluation-on-pascal-voc-2007). Afterwards, unzip `VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt.zip`, and place the contents (*.ckpt.index and *.ckpt.data-00000-of-00001) under `SSD/checkpoints`.

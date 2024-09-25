## Training and testing
1. Training datasets.
(1) Make your own dataset by labelimg, put the original image in VOCdevkitVOC2007JPEGImages, and put the label file in VOCdevkitVOC2007Annotations, run the voc_annotation.py to distinguish between the training set and the test set, and finally run the train_faster-rcnn.py training. 
(2) Run crop.py and select Batch Crop Images in the frnn.py, and use faster-rcnn to crop out all the cells in one folder in batches.()(3)Use labelme to label the cropped cells, then put them in the datasetsbefore folder, run json_to_dataset.py to change the labels in jion format to png format, and store the original image in datasetsJPEGImages, and store the label file in datasetsSegmentationClass. Transfer JPEGImages and SegmentationClass to VOCdevkit_unetVOC2007, run voc_annotation_u-net.py divide the training set and dataset, and run the train_u-net.py to train the U-Net model.
2. Perform Cell image detection
predict.py run, enter img/1.jpg  



## Additional Notes
1.The trained file will be saved in logs/, remember to change the file name to the corresponding file name during the test
2.The part where u-net is embedded in faster-rcnn is reflected in the frcnn.py
3.The combination of template matching and combined networks is reflected in the frcnn.py





## Reference
https://github.com/chenyuntc/simple-faster-rcnn-pytorch
https://github.com/ggyyzm/pytorch_segmentation    
https://github.com/bubbliiiing/faster-rcnn-pytorch/tree/bilibili
https://github.com/bubbliiiing/unet-pytorch/tree/bilibili
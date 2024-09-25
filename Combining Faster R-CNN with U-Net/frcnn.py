import colorsys
import os
import time
import math 
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageEnhance  

from nets.frcnn import FasterRCNN
from utils.utils import (cvtColor, get_classes, get_new_img_size, resize_image,
                         preprocess_input, show_config)
from utils.utils_bbox import DecodeBox
import csv
from unet import Unet
import pandas as pd
import random
unet = Unet()    
name_classes    = ["background","u87"]   

#----------------------------------------------------------------------------------------#
#To predict with your own trained model, you need to modify 2 parameters:
#model_path and classes_path both need to be changed!
#----------------------------------------------------------------------------------------#
class FRCNN(object):
    _defaults = {
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        #   Be sure to modify the model_path and classes_path when using your own trained model for prediction!
        #   The model_path should point to the weight file in the logs folder, and the classes_path should point to the txt file in the model_data folder.
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------#
        "model_path"    : 'logs/Faster_rcnn.pth.pth',
        "classes_path"  : 'model_data/new_classes.txt',
        #-----------------------------------------------------------------------------------------------------------------#
        #   The main body of the network for feature extraction, ResNet50 or VGG.
        #-----------------------------------------------------------------------------------------------------------------#
        "backbone"      : "resnet50",
        #-----------------------------------------------------------------------------------------------------------------#
        #   Prediction boxes with scores higher than the confidence threshold will be retained.
        #-----------------------------------------------------------------------------------------------------------------#
        "confidence"    : 0.9,
        #-----------------------------------------------------------------------------------------------------------------#
        #   The size of the nms_iou used in Non-Maximum Suppression (NMS).
        #-----------------------------------------------------------------------------------------------------------------#
        "nms_iou"       : 0.03,
        #-----------------------------------------------------------------------------------------------------------------#
        #   Used to specify the size of the anchor boxes.
        #-----------------------------------------------------------------------------------------------------------------#
        'anchors_size'  : [8, 16, 32],
        #---------------------------------------------------------------------------#
        #   Whether to use Cuda for translation.
        #---------------------------------------------------------------------------#
        "cuda"          : True,
        #-------------------------------#
        #  Crop and save in bulk : 0
        #  Save only one         : 1
        #-------------------------------#        
        "Batch_cropping"      : 0,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   initialization faster RCNN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
 
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        self.std    = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std    = self.std.cuda()
        self.bbox_util  = DecodeBox(self.std, self.num_classes)

        #---------------------------------------------------#
        #   Set different colors for drawing boxes
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #  Loading a model
    #---------------------------------------------------#
    def generate(self):

        self.net    = FasterRCNN(self.num_classes, "predict", anchor_scales = self.anchors_size, backbone = self.backbone)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    #---------------------------------------------------#
    #   Detect images
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False):

        image_shape = np.array(np.shape(image)[0:2])

        input_shape = get_new_img_size(image_shape[0], image_shape[1])

        image       = cvtColor(image)

        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            

            roi_cls_locs, roi_scores, rois, _ = self.net(images)

            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
         
            if len(results[0]) <= 0:
                return image
                
            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        #---------------------------------------------------------#
        #   Set font and border thickness
        #---------------------------------------------------------#
   
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(2.0e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 0.5))
        #---------------------------------------------------------#
        # count 
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   Whether to crop the target
        #---------------------------------------------------------#
        if crop:
            mean_x = []
            mean_y = []
            for i, c in list(enumerate(top_label)):
                predicted_class = self.class_names[int(c)]
                if predicted_class == 'Sphere_Probe':
                    continue
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))
                

                img_origin_crop = "img_origin_crop"
                if not os.path.exists(img_origin_crop):
                    os.makedirs(img_origin_crop)

                dir_uimage_save_path = "dir_uimage_save_path"
                if not os.path.exists(dir_uimage_save_path):
                    os.makedirs(dir_uimage_save_path)
    

                                    
                if predicted_class == 'u87':
                    #-----------------------------------------------------------------------------------------------------------------#
                    # Whether to crop the dataset in batches
                    #-----------------------------------------------------------------------------------------------------------------#  
                    if self.Batch_cropping == 0:
                        random_integer = random.randint(1, 100000000000)
                        a=random_integer
                        crop_image_origin = image.crop([left, top, right, bottom])                   
                        crop_image_origin.save(os.path.join(img_origin_crop, "crop_" +str(a)+ "_"+str(i) + ".jpg"), quality=95, subsampling=0)
                    elif self.Batch_cropping ==1:                   
                        crop_image_origin = image.crop([left, top, right, bottom])                                      
                        crop_image_origin.save(os.path.join(img_origin_crop, "crop_"+ "_"+str(i) + ".jpg"), quality=95, subsampling=0)
#----------------------------------------------------------------------------------------------------------------#
#                               Embed U-Net network      
#----------------------------------------------------------------------------------------------------------------#                   
                    uimage = unet.detect_image(crop_image_origin,name_classes=name_classes)
                    uimage_32 = np.float32(uimage)
                    uimage = np.uint8(uimage_32)
                    cv2.imwrite(os.path.join(dir_uimage_save_path, "crop_" + str(i) + ".png"), uimage)
                    print("save uimage crop_" + str(i) + ".png to " + dir_uimage_save_path)
#----------------------------------------------------------------------------------------------------------------#
#                             Calculate the centroid of the cell    
#----------------------------------------------------------------------------------------------------------------#                                        

                    crop_image = cv2.cvtColor(uimage, cv2.COLOR_BGR2GRAY)
                    _, binary_img = cv2.threshold(crop_image, 200, 255, cv2.THRESH_BINARY)      
                    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      
                    img_pil = Image.fromarray(cv2.cvtColor(binary_img, cv2.COLOR_BGR2RGB))  
                    draw = ImageDraw.Draw(img_pil) 
                      
                    for i, cnt in enumerate(contours):  
  
                        M = cv2.moments(cnt)  
                 
                        if M["m00"] != 0:  
                            mean_xs = int(M["m10"] / M["m00"])  
                            mean_ys = int(M["m01"] / M["m00"])
                              
                    mean_x.append(mean_xs)       
                    mean_y.append(mean_ys)                      
                    print('mean_x',mean_x)
                    print('mean_y',mean_y)
                                       
                else:
                    center_x_ball = (right - left) / 2
                    center_y_ball = (bottom - top) / 2
                    mean_y.append(center_y_ball)
                    mean_x.append(center_x_ball)
                    continue


        # ---------------------------------------------------------#
        #   Image drawing
        # ---------------------------------------------------------#
        

        list_class=[]
        list_x = []
        list_y = []
        list_class.clear()
        list_x.clear()
        list_y.clear()

      
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
             
            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))



        # ---------------------------------------------------------#
        #   The position of the cell center in the original image
        # ---------------------------------------------------------#

            center_x = mean_x[i]+left#
            center_y = mean_y[i]+top#                     
            print('center_x',center_x)
            print('center_y',center_y)
#----------------------------------------------------------------------------------------------------------------#
#       Obtain relative coordinates from the coordinates acquired through template matching   
#----------------------------------------------------------------------------------------------------------------#   
            df = pd.read_csv('center_coordinates.csv')
            tx = df['x'].astype(int)
            ty = df['y'].astype(int)          
            x1 = -(center_x-tx)
            y1 = -(center_y-ty)
            
            x1 = np.around(x1, decimals=1).item()
            y1 = np.around(y1, decimals=1).item()
#            label = f'{predicted_class}:{score:.2f}  {"num:"} {i} \n {"Relative:"} {(y1,x1)}'
            label = f''

            label_number = '{}'.format(i)
            class_detect = '{} '.format(predicted_class)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            number_size = draw.textsize(label_number, font)

            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
            else:
                    text_origin = np.array([left, top + 1])

            if top - number_size[0] >= 0:
                    number_s = np.array([left, top - number_size[0]-16])#4
            else:
                    number_s = np.array([left, top + 1])

            for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])


 
            class_data = 'Cell class：{}'.format(class_detect)
            list_class.append(class_data)
             
            for i in range(thickness):
                    draw.rectangle([center_x + 3, center_y + 15, center_x - 3, center_y - 15], fill=self.colors[c], outline=self.colors[c])
                    draw.rectangle([center_x + 15, center_y + 3, center_x - 15, center_y - 3],
                                   fill=self.colors[c], outline=self.colors[c])
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
#                            
            blue_color = (0, 0, 255)
            green_color = (0, 255, 0)                   
            draw.rectangle([tx + 5, ty + 20, tx - 5, ty - 20], fill=self.colors[c], outline=blue_color)
            draw.rectangle([tx + 20, ty+ 5, tx - 20, ty- 5],
                                   fill=self.colors[c], outline=blue_color)
            draw.rectangle([tx + 40, ty+ 40, tx - 40, ty- 40], outline=green_color)
            draw.rectangle([tx + 41, ty+ 41, tx - 41, ty- 41], outline=green_color)   
            draw.rectangle([tx + 42, ty+ 42, tx - 42, ty- 42], outline=green_color)
            draw.rectangle([tx + 43, ty+ 43, tx - 43, ty- 43], outline=green_color)
                
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image, top_label, count

 
    def get_FPS(self, image, test_interval):

        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])

        image       = cvtColor(image)
        
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)

            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                roi_cls_locs, roi_scores, rois, _ = self.net(images)

                results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                        nms_iou = self.nms_iou, confidence = self.confidence)
                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #---------------------------------------------------#
    #   Detect the picture
    #---------------------------------------------------#
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")

        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])

        image       = cvtColor(image)

        image_data  = resize_image(image, [input_shape[1], input_shape[0]])

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)

            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)

            if len(results[0]) <= 0:
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
    
    
    
    
    


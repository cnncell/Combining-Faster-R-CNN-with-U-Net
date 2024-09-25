#-------------------------------------#
#       Train on the dataset
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import (get_classes, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    #-------------------------------#
    # Whether to use Cuda
    # If you don't have a GPU, you can set it to False
    #------------------------------- #
    Cuda            = True
    #----------------------------------------------#
    # Seed is used to fix random seeds
    # So that you can get the same results every time you train independently
    #---------------------------------------------- #
    seed            = 11
    #---------------------------------------------------------------------#
    # train_gpu GPU used for training
    # The default is the first card, the double card is [0, 1], and the triple card is [0, 1, 2]
    # When using multiple GPUs, the batch on each card is the total batch divided by the number of cards.
    #--------------------------------------------------------------------- #
    train_gpu       = [0,]
    #---------------------------------------------------------------------#
    # Whether fp16 uses mixed-precision training
    # It can reduce the video memory by about half, and requires pytorch 1.7.1 or higher
    #--------------------------------------------------------------------- #
    fp16            = False
    #---------------------------------------------------------------------#
    # classes_path points to the txt under the model_data, which is related to the dataset you trained 
    # Be sure to modify the classes_path before training to make it correspond to your own dataset
    #--------------------------------------------------------------------- #
    classes_path    = 'model_data/voc_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    # The pre-trained weights of the model are common to different datasets because the features are common.
    #---------------------------------------------------------------------------------------------------------------------------- #
    model_path      = 'model_data/voc_weights_resnet.pth'
    #------------------------------------------------------#
    # input_shape Enter the shape size
    #------------------------------------------------------ #
    input_shape     = [600, 600]
    #---------------------------------------------#
    #   vgg
    #   resnet50
    #---------------------------------------------#
    backbone        = "resnet50"
    #----------------------------------------------------------------------------------------------------------------------------#
    # Whether pretrained uses the pre-trained weights of the backbone network, the weights of the backbone are used here, so they are loaded when the model is built.
    # If model_path is set, the weight of the trunk does not need to be loaded, and the value of pretrained is meaningless.
    # If you do not set the model_path, pretrained = True, only the trunk will be loaded to start training.
    # If you do not set model_path, pretrained = False, Freeze_Train = Fasle, the training starts from 0 and there is no process of freezing the trunk.
    #---------------------------------------------------------------------------------------------------------------------------- #
    pretrained      = False
    #------------------------------------------------------------------------#
    # anchors_size is used to set the size of the prior box, and there are 9 prior boxes for each feature point.
    # anchors_size Each number corresponds to 3 prior boxes.
    # When anchors_size = [8, 16, 32], the width and height of the generated prior box are approximately:
    #   [90, 180] ; [180, 360]; [360, 720]; [128, 128]; 
    #   [256, 256]; [512, 512]; [180, 90] ; [360, 180]; 
    #   [720, 360]; For more information, see anchors.py
    # If you want to detect small objects, you can reduce the number of anchors_size in front.
    # e.g. set anchors_size = [4, 16, 32]
    #------------------------------------------------------------------------ #
    anchors_size    = [8, 16, 32]
    
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 4
    
    UnFreeze_Epoch      = 800
    Unfreeze_batch_size = 2
    Freeze_Train        = True    
    #------------------------------------------------------------------#
    # Init_lr The maximum learning rate of the model
    # When using the Adam optimizer, it is recommended to set Init_lr=1e-4
    # When using the SGD optimizer, it is recommended to set Init_lr=1e-2
    # Min_lr The minimum learning rate of the model, which defaults to 0.01 of the maximum learning rate
    #------------------------------------------------------------------ #
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    # optimizer_type The types of optimizers used are ADAM and SGD
    # When using the Adam optimizer, it is recommended to set Init_lr=1e-4
    # When using the SGD optimizer, it is recommended to set Init_lr=1e-2
    # The momentum parameter used inside the momentum optimizer
    # weight_decay Weight decay to prevent overfitting
    # ADAM will cause weight_decay error, and it is recommended to set it to 0 when using ADAM.
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    # lr_decay_type Use the learning rate reduction method, optional 'step', 'cos'
    #------------------------------------------------------------------ #
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    # save_period How many epochs save the weights once
    #------------------------------------------------------------------ #
    save_period         = 5
    #------------------------------------------------------------------#
    # save_dir The folder where the weights and log files are saved
    #------------------------------------------------------------------ #
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    # eval_flag Whether to evaluate at training time, which is the validation set
    # After installing the pycocotools library, the evaluation experience is better.
    # eval_period represents how many epochs are evaluated at one time, and frequent evaluations are not recommended
    # Evaluations take a lot of time, and frequent evaluations can lead to very slow training
    # The mAP obtained here will be different from the one obtained by get_map.py for two reasons:
    # (1) The mAP obtained here is the mAP of the verification set.
    # (2) The evaluation parameters set here are conservative in order to speed up the evaluation.
    #------------------------------------------------------------------ #
    eval_flag           = True
    eval_period         = 5
    #------------------------------------------------------------------#
    # num_workers Sets whether to use multithreading to read data, 1 indicates that multithreading is disabled
    # When enabled, it will speed up data reading, but it will take up more memory
    # Enable multi-threading when IO is the bottleneck, that is, the GPU computing speed is much faster than the speed of reading images.
    #------------------------------------------------------------------ #
    num_workers         = 4
    #----------------------------------------------------#
    # Get the image path and labels
    #---------------------------------------------------- #
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'
    
    #----------------------------------------------------#
    # Get classes and anchors
    #---------------------------------------------------- #
    class_names, num_classes = get_classes(classes_path)

    #------------------------------------------------------#
    # Set up the graphics card you want to use
    #------------------------------------------------------ #
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))
    seed_everything(seed)
    
    model = FasterRCNN(num_classes, anchor_scales = anchors_size, backbone = backbone, pretrained = pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':

        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m reminder that it is normal for the head part to not be loaded, and it is wrong for the backbone part to not be loaded.\033[0m")

    #----------------------#
    #   记录Loss
    #----------------------#
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)


    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()


    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    
    show_config(
        classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('The dataset is too small to be trained, so enrich the dataset.')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training step size above %d.\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] The total amount of training data in this run is %d, the Unfreeze_batch_size is %d, a total of %d epochs are trained, and the total training step size is %d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] Since the total training step size is %d, which is less than the recommended total step size %d, it is recommended to set the total generation to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    # The backbone feature extraction network features are universal, and freezing training can speed up the training
    # It can also prevent the metric from being destroyed in the early stage of training.
    # Init_Epoch is the starting generation
    # Freeze_Epoch for the generation of frozen training
    # UnFreeze_Epoch total training generation
    # If OOM or video memory is insufficient, please reduce the Batch_size
    #------------------------------------------------------ #
    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.extractor.parameters():
                param.requires_grad = False

        model.freeze_bn()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training, so expand the dataset.")

        train_dataset   = FRCNNDataset(train_lines, input_shape, train = True)
        val_dataset     = FRCNNDataset(val_lines, input_shape, train = False)

        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=frcnn_dataset_collate, 
                                    worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=frcnn_dataset_collate, 
                                    worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

        train_util      = FasterRCNNTrainer(model_train, optimizer)

        eval_callback   = EvalCallback(model_train, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                        eval_flag=eval_flag, period=eval_period)


        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.extractor.parameters():
                    param.requires_grad = True

                model.freeze_bn()

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training, so expand the dataset.")

                gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=frcnn_dataset_collate, 
                                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=frcnn_dataset_collate, 
                                            worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

                UnFreeze_flag = True
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir)
            
        loss_history.writer.close()
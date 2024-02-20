'''
[USAGE]
python train.py --gpu_id [insert gpu id] --bs 4 --epochs 700\
--lr 0.0001 --checkpoint_path ./checkpoints/ --log_path ./logs/ --data_path [dataset path(images)] --label_path [dataset path(labels)]

e.g) 데이터셋 : argoverse dataset
python train.py --gpu_id 2 --bs 4 --epochs 700 --lr 0.0001 --checkpoint_path ./checkpoints/ --log_path ./logs/ --data_path /Data1/hm/DRAEM_TRAIN_DATASET/argoverse/images --label_path /Data1/hm/DRAEM_TRAIN_DATASET/argoverse/labels
           
'''
import torch
from torch.utils.data import DataLoader
from patch.utils.dataset import YOLODataset
from patch.utils.patch import PatchApplier, PatchTransformer
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_od_shield.model import ReconstructiveSubNetwork
from datetime import datetime, timedelta

from loss.loss import FocalLoss, SSIM
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn
import time
from glob import glob
from tqdm import tqdm
from datetime import datetime
import pytz
import warnings
import torch.nn as nn
warnings.filterwarnings('ignore')

korea_time = 'Asia/Seoul'
tz = pytz.timezone(korea_time)

def read_image(path) -> torch.Tensor:
    """
    Read an input image to be used as a patch

    Arguments:
        path: Path to the image to be read.
    """
    patch_img = Image.open(path).convert("RGB")
    adv_patch_cpu = T.ToTensor()(patch_img)
    return adv_patch_cpu


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'conv'):
            nn.init.normal_(m.conv.weight, 0.0, 0.02)
            if hasattr(m.conv, 'bias') and m.conv.bias is not None:
                nn.init.constant_(m.conv.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'bn') and m.bn is not None:
            nn.init.normal_(m.bn.weight, 1.0, 0.02)
            nn.init.constant_(m.bn.bias, 0)
            
            
def train_on_device(args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    cur_time = datetime.now(tz)
    run_name = cur_time.strftime("%Y%m%d_%H%M")
    
    abspath = os.path.abspath(os.getcwd())+'/'  
    
    print(' path to save pretrained model : ',abspath,os.path.join(args.checkpoint_path, f'{run_name}'))
    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.cuda()
    model.apply(weights_init)
    
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)
    
    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    
    print('####################################################')
    print('#############   [LOSS] : L2         loss ###########')
    print('#############   [LOSS] : SSIM       loss ###########')    
    print('#############   [LOSS] : Perceptual loss ###########')
    print('####################################################')    

    dataset = YOLODataset(image_dir = args.data_path,
                          label_dir = args.label_path, 
                          max_labels = 48,
                          model_in_sz = [640, 640],
                          shuffle = True)

    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    patch_dirs = glob(os.path.join(args.patch_paths, '*.png'))
    
    savelog_per_step_by_batchsize = ((len(dataloader) // args.bs)*args.epochs) // 650
    n_iter = 0
    args.visualize = True
    
    for epoch in range(args.epochs):
        print("Epoch: "+str(epoch))
        
        for i_batch, (sample_batched, label_batched) in enumerate(tqdm(dataloader)):
            
            patch = random.choice(patch_dirs)
            
            adv_patch_cpu = read_image(path=patch)
            
            patch_transformer = PatchTransformer([0.25, 0.4], [0.5, 0.8], 0.1, [-0.25, 0.25], [-0.25, 0.25], torch.device("cuda")).cuda()
            patch_applier = PatchApplier(1).cuda()                

            gray_batch = sample_batched.cuda()
            label_batch = label_batched.cuda()
            adv_patch = adv_patch_cpu.cuda()
            adv_batch_t = patch_transformer(adv_patch, label_batch,
                                            model_in_sz=[640, 640],
                                            use_mul_add_gau="all",
                                            do_transforms=True,
                                            do_rotate=True,
                                            rand_loc=True)
            p_img_batch = patch_applier(gray_batch, adv_batch_t)
            aug_gray_batch = F.interpolate(p_img_batch, (640, 640))
            
            gray_rec, perceptual_loss = model(aug_gray_batch, gray_batch)

            l2_loss = loss_l2(gray_rec, gray_batch)
            ssim_loss = loss_ssim(gray_rec, gray_batch)                 
            loss = l2_loss + ssim_loss + perceptual_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.visualize and n_iter % savelog_per_step_by_batchsize == 0:
                visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                visualizer.plot_loss(perceptual_loss, n_iter, loss_name='perceptual_loss')
                visualizer.plot_loss(loss, n_iter, loss_name='loss')

            n_iter +=1
            
            del patch, adv_patch_cpu, patch_transformer, patch_applier
            
        scheduler.step()
        
        if (epoch+1) % 50 == 0 or epoch == 0 :
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f'{run_name}_{epoch+1}.pckl')) 

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--bs', action='store', type=int, default = 2, required=False)
    parser.add_argument('--lr', action='store', type=float, default = 0.0001, required=False)
    parser.add_argument('--epochs', action='store', type=int, default = 700, required=False)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    
    parser.add_argument('--checkpoint_path', action='store', type=str, default = './checkpoints/', required=False)
    parser.add_argument('--log_path', action='store', type=str, default = './logs/', required=False)
    parser.add_argument('--visualize', action='store_true')
    
    parser.add_argument('--data_path', default='/Data1/hm/OD_SHIELD/datasets/argoverse/train/images') # modify the dataset images path
    parser.add_argument('--label_path', default='/Data1/hm/OD_SHIELD/datasets/argoverse/train/labels') # modify the dataset labels path
    parser.add_argument('--patch_paths', default='/Data1/hm/OD_SHIELD/patch/patch_sample') 

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        train_on_device(args)


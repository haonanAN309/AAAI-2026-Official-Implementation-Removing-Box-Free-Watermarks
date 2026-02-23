import torch
import torch.nn as nn
from models.HidingUNet import UnetGenerator
from models.HidingRes import HidingRes
from models.HidingENet import HidingENet
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import os
from data.ImageFolderDataset import MyImageFolder
import torchvision.transforms as trans
import utils.transformed as transforms
import torchvision.utils as vutils

def save_result_pic(this_batch_size, originalLabelvA, originalLabelvB, Container_allImg, uncover_img, secretLabelv, RevSecImg,
                    origin_rev_secret_img, epoch, i, save_path):
    originalFramesA = originalLabelvA.resize_(this_batch_size, 3, 256, 256)
    originalFramesB = originalLabelvB.resize_(this_batch_size, 3, 256, 256)
    container_allFrames = Container_allImg.resize_(this_batch_size, 3, 256, 256)
    uncoverFrames = uncover_img.resize_(this_batch_size, 3, 256, 256)
    secretFrames = secretLabelv.resize_(this_batch_size, 3, 256, 256)
    revSecFrames = RevSecImg.resize_(this_batch_size, 3, 256, 256)
    originrevSecFrames = origin_rev_secret_img.resize_(this_batch_size, 3, 256, 256)
    
    showResult = torch.cat(
        [originalFramesA, originalFramesB,  container_allFrames, uncoverFrames, secretFrames,
         revSecFrames, originrevSecFrames], 0)

    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    vutils.save_image(showResult, resultImgName, nrow=this_batch_size, padding=1, normalize=False)

# Hnet and Enet of victim two
Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
Hnet.cuda()
Enet = HidingENet()
Enet.cuda()

# FHnet loading
FHnet = UnetGenerator(input_nc=3, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
FHnet.cuda()

# fix Hnet, Enet, FHnet
for param in Hnet.parameters():
    param.requires_grad = False

for param in Enet.parameters():
    param.requires_grad = False
    
for param in FHnet.parameters():
    param.requires_grad = False

# parameters loading
Hnet.load_state_dict(torch.load(""))
Enet.load_state_dict(torch.load(""))
FHnet.load_state_dict(torch.load(""))


FHnet.eval()
Hnet.eval()
Enet.eval()

with torch.no_grad():
    testdir = '/home/user2/ahn/dataset/stable_diffusion_generated/dataset_a_b/test'
    test_dataset = MyImageFolder(
    testdir,
    transforms.Compose([
        # trans.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]))
    test_loader = DataLoader(test_dataset, batch_size=16,
                              shuffle=True, num_workers=int(8))
    
    
    Tensor = torch.cuda.FloatTensor

    loader = transforms.Compose([
        # trans.Grayscale(num_output_channels=1),
        transforms.ToTensor(),])
    clean_img = Image.open("secret/clean.png")  # 256 * 256
    clean_img = loader(clean_img)
    secret_img = Image.open("secret/flower.png")  # 256 * 256
    secret_img = loader(secret_img)
    key = torch.load("key.pt")

    for i, data in enumerate(test_loader, 0):
        this_batch_size = int(data.size()[0])
        cover_img = data[0:this_batch_size, :, :, :]
        cover_img_A = cover_img[:, :, 0:256, 0:256]  # divide cover_img into two parts
        cover_img_B = cover_img[:, :, 0:256, 256:512]

        secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)  # repeat batch_size's times
        secret_img = secret_img[0:this_batch_size, :, :, :]

        clean_img = clean_img.repeat(this_batch_size, 1, 1, 1)  # repeat batch_size's times
        clean_img = clean_img[0:this_batch_size, :, :, :]
        
        key = key.repeat(this_batch_size, 1, 1, 1)
        key = key[0:this_batch_size, :, :, :]

        if torch.cuda.is_available():
            cover_img = cover_img.cuda()
            cover_img_A = cover_img_A.cuda()
            cover_img_B = cover_img_B.cuda()
            secret_img = secret_img.cuda()
            clean_img = clean_img.cuda()
            key = key.cuda()
        
        concat_img = torch.cat([cover_img_B, secret_img], dim=1)
        container_img = Hnet(concat_img) # b'
        
        origin_rev_secret_img = Enet(container_img, key)
        recover_B = cover_img_A - (FHnet(cover_img_A) - container_img)
        rev_secret_img = Enet(recover_B, key)
        # FHnet_concate_img = torch.cat([container_img, clean_img], dim=1)
        # uncover_img = FHnet(FHnet_concate_img)
        # rev_secret_img = Enet(uncover_img)

        output_dir = ""
        save_result_pic(this_batch_size, cover_img_A, cover_img_B, container_img, recover_B,
                            secret_img, origin_rev_secret_img, rev_secret_img,
                            1, i, output_dir)
        

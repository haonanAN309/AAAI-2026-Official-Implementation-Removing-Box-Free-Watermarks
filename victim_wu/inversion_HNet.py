import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import utils.transformed as transforms
from models.HidingUNet import UnetGenerator
from models.HidingENet import HidingENet
from data.ImageFolderDataset import MyImageFolder
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Enet', default='',
                    help="path to Extractnet (to continue training)")
parser.add_argument('--IHnet', default='',
                    help="path to SurrogateNet (to continue training)")
parser.add_argument('--trainpics', default='output_inverse_HNet_RGB/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='output_inverse_HNet_RGB/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='output_inverse_HNet_RGB/',
                    help='folder to output test images')
parser.add_argument('--runfolder', default='output_inverse_HNet_RGB/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='output_inverse_HNet_RGB/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='output_inverse_HNet_RGB/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='output_inverse_HNet_RGB/',
                    help='folder to save the experiment codes')

parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')
parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')


#datasets to train
parser.add_argument('--datasets', type=str, default='',
                    help='denoise/derain')

#read secret image
parser.add_argument('--secret', type=str, default='flower',
                    help='secret folder')

#hyperparameter of loss
parser.add_argument('--beta', type=float, default=1,
                    help='hyper parameter of beta :secret_reveal err')
parser.add_argument('--betamse', type=float, default=10000,
                    help='hyper parameter of beta: mse_loss')
parser.add_argument('--betaconsist', type=float, default=1,
                    help='hyper parameter of beta: consist_loss')
parser.add_argument('--betapixel', type=float, default=100,
                    help='hyper parameter of beta :pixel_loss weight')
parser.add_argument('--alphaA', type=float, default=0.2,
                   help='hyper parameter of alpha: imgA')
parser.add_argument('--alphacoverB', type=float, default=0.8,
                   help='hyper parameter of alpha: covered B')
parser.add_argument('--num_downs', type=int, default= 7 , help='nums of  Unet downsample')
parser.add_argument('--clip', action='store_true', help='clip container_img')


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ############### define global parameters ###############
    global opt, optimizerIHnet, writer, logPath, schedulerIHnet, val_loader
    global criterion_pixelwise, mse_loss, pixel_loss, smallestLoss

    opt = parser.parse_args()
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True


    ############  create the dirs to save the result #############

    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    experiment_dir = opt.hostname + "_" + opt.remark + "_" + cur_time
    opt.outckpts += experiment_dir + "/checkPoints"
    opt.trainpics += experiment_dir + "/trainPics"
    opt.validationpics += experiment_dir + "/validationPics"
    opt.outlogs += experiment_dir + "/trainingLogs"
    opt.outcodes += experiment_dir + "/codes"
    opt.testPics += experiment_dir + "/testPics"
    opt.runfolder += experiment_dir + "/run"

    if not os.path.exists(opt.outckpts):
        os.makedirs(opt.outckpts)
    if not os.path.exists(opt.trainpics):
        os.makedirs(opt.trainpics)
    if not os.path.exists(opt.validationpics):
        os.makedirs(opt.validationpics)
    if not os.path.exists(opt.outlogs):
        os.makedirs(opt.outlogs)
    if not os.path.exists(opt.outcodes):
        os.makedirs(opt.outcodes)
    if not os.path.exists(opt.runfolder):
        os.makedirs(opt.runfolder)
    if (not os.path.exists(opt.testPics)) and opt.test != '':
        os.makedirs(opt.testPics)

    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    print_log(str(opt), logPath)
    save_current_codes(opt.outcodes)
    # tensorboardX writer
    writer = SummaryWriter(log_dir=opt.runfolder, comment='**' + opt.hostname + "_" + opt.remark)

    DATA_DIR = opt.datasets
    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'val')

    train_dataset = MyImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize((256, 256)),
            # trans.Grayscale(num_output_channels=1),
            transforms.ToTensor(),

        ]))
    val_dataset = MyImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((256, 256)),
            # trans.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]))

    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                              shuffle=True, num_workers=int(opt.workers))

    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                            shuffle=False, num_workers=int(opt.workers))

    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=opt.num_downs, output_function=nn.Sigmoid)
    Hnet.cuda()
    
    IHnet = UnetGenerator(input_nc=3, output_nc=3, num_downs= opt.num_downs, output_function=nn.Sigmoid)
    IHnet.cuda()
    IHnet.apply(weights_init)

    Enet = HidingENet()
    Enet.cuda()

    # setup optimizer
    optimizerIHnet = optim.Adam(IHnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    schedulerIHnet = ReduceLROnPlateau(optimizerIHnet, mode='min', factor=0.2, patience=5, verbose=True)

    if opt.Hnet != '':
        Hnet.load_state_dict(torch.load(opt.Hnet))
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()
    print_network(Hnet)
    
    if opt.IHnet != '':
        IHnet.load_state_dict(torch.load(opt.IHnet))
    if opt.ngpu > 1:
        IHnet = torch.nn.DataParallel(IHnet).cuda()
    print_network(IHnet)

    if opt.Enet != '':
        Enet.load_state_dict(torch.load(opt.Enet))
    if opt.ngpu > 1:
        Enet = torch.nn.DataParallel(Enet).cuda()
    print_network(Enet)

    for param in Hnet.parameters():
        param.requires_grad = False
    for param in Enet.parameters():
        param.requires_grad = False
        
    # define loss
    mse_loss = nn.MSELoss().cuda()

    smallestLoss = 10000
    print_log("training is beginning .......................................................", logPath)
    for epoch in range(opt.niter):
        ######################## train ##########################################
        train(train_loader, epoch, Hnet=Hnet, IHnet=IHnet, Enet=Enet)

        ####################### validation  #####################################
        val_sumloss= validation(val_loader,  epoch, Hnet=Hnet, IHnet=IHnet,  Enet=Enet)

        ####################### adjust learning rate ############################
        schedulerIHnet.step(val_sumloss)

        # save the best model parameters
        if val_sumloss < globals()["smallestLoss"]:
            globals()["smallestLoss"] = val_sumloss
            # for parallel training: IHnet.module.state_dict()
            torch.save(IHnet.module.state_dict(),
                       '%s/IHnet_epoch_%d,sumloss=%.6f.pth' % (
                           opt.outckpts, epoch, val_sumloss))
    writer.close()


def train(train_loader, epoch, Hnet, IHnet, Enet):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    IHnetlosses = AverageMeter()

    # switch to train mode
    Hnet.eval()
    IHnet.train()
    Enet.eval()
    # Tensor type
    Tensor = torch.cuda.FloatTensor
    loader = transforms.Compose([transforms.ToTensor(),])
    secret_img = Image.open("secret/flower.png")  # 256 * 256
    secret_img = loader(secret_img)
    key = torch.load('key.pt')
    
    start_time = time.time()
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        IHnet.zero_grad()

        this_batch_size = int(data.size()[0])
        cover_img_B = data[0:this_batch_size, :, :, :]
        secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)  # repeat batch_size's times
        secret_img = secret_img[0:this_batch_size, :, :, :]
        key = key.repeat(this_batch_size, 1, 1, 1)
        key = key[0:this_batch_size, :, :, :]
        
        if opt.cuda:
            cover_img_B = cover_img_B.cuda()
            secret_img = secret_img.cuda()
            key = key.cuda()

        cover_img_Bv = Variable(cover_img_B)
        keyv = Variable(key)
        concat_img = torch.cat([cover_img_B, secret_img], dim=1)
        container_img = Hnet(concat_img)  
        containerv = Variable(container_img)

        generated_image = IHnet(containerv)
        uncover_img_B = Enet(cover_img_B, keyv)
        uncover_img = Enet(containerv, keyv)
        uncover_img_generated = Enet(generated_image, keyv)

        err_IHnet = opt.betamse * mse_loss(generated_image, cover_img_Bv)

        err_IHnet.backward()
        optimizerIHnet.step()

        IHnetlosses.update(err_IHnet.data, this_batch_size)

        batch_time.update(time.time() - start_time)
        start_time = time.time()
        # log writing
        log = '[%d/%d][%d/%d]\t Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.niter, i, len(train_loader),
               IHnetlosses.val, data_time.val, batch_time.val)

        if i % opt.logFrequency == 0:
            print_log(log, logPath)
        else:
            print_log(log, logPath, console=False)

        if epoch % 1 == 0 and i % opt.resultPicFrequency == 0:
            save_result_pic(this_batch_size, cover_img_Bv.data, containerv.data, generated_image.data, uncover_img_B.data,
                                            uncover_img.data, uncover_img_generated.data,
                                            epoch, i, opt.trainpics)

    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerIHnet_lr = %.8f" % (
        optimizerIHnet.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_sumloss=%.6f" % (IHnetlosses.avg)

    print_log(epoch_log, logPath)

    writer.add_scalar("lr/H_lr", optimizerIHnet.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/beta", opt.beta, epoch)
    writer.add_scalar('train/Sum_loss', IHnetlosses.avg, epoch)


def validation(val_loader, epoch, Hnet, IHnet, Enet):
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    IHnet.eval()
    Enet.eval()

    IHnetlosses = AverageMeter()
    loader = transforms.Compose([transforms.ToTensor(),])
    secret_img = Image.open("secret/flower.png")  # 256 * 256
    secret_img = loader(secret_img)
    key = torch.load('key.pt')
    # Tensor type
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):

            this_batch_size = int(data.size()[0])
            cover_img_B = data[0:this_batch_size, :, :, :]
            secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)  # repeat batch_size's times
            secret_img = secret_img[0:this_batch_size, :, :, :]
            key = key.repeat(this_batch_size, 1, 1, 1)
            key = key[0:this_batch_size, :, :, :]

            if opt.cuda:
                cover_img_B = cover_img_B.cuda()
                secret_img = secret_img.cuda()
                key = key.cuda()

            cover_img_Bv = Variable(cover_img_B)
            keyv = Variable(key)
            concat_img = torch.cat([cover_img_B, secret_img], dim=1)
            container_img = Hnet(concat_img)  
            containerv = Variable(container_img)

            generated_image = IHnet(containerv)
            uncover_img_B = Enet(cover_img_B, keyv)
            uncover_img = Enet(containerv, keyv)
            uncover_img_generated = Enet(generated_image, keyv)
            err_IHnet = opt.betamse * mse_loss(generated_image, cover_img_Bv)

            IHnetlosses.update(err_IHnet.data, this_batch_size)

            if i % 50 == 0:
                save_result_pic(this_batch_size, cover_img_Bv.data, containerv.data, generated_image, 
                                uncover_img_B.data, uncover_img.data, uncover_img_generated.data,
                                epoch, i, opt.validationpics)


    val_time = time.time() - start_time
    val_log = "validation[%d] val_sumloss = %.6f\t validation time=%.2f" % (
        epoch, IHnetlosses.avg, val_time)

    print_log(val_log, logPath)

    writer.add_scalar('validation/Sum_loss', IHnetlosses.avg, epoch)

    print(
        "#################################################### validation end ########################################################")

    return IHnetlosses.avg


# custom weights initialization called on netG and netD
# these initializations are often used in GAN.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)
    cur_work_dir, mainfile = os.path.split(main_file_path)

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)

# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)

def save_result_pic(this_batch_size, originalLabelvB, originalLabelvContainer, generated, uncover_img_B, uncover_img, uncover_img_generate, epoch, i, save_path):
    originalFramesB = originalLabelvB.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
    originalFramesContainer = originalLabelvContainer.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
    generated = generated.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
    uncover_img_B = uncover_img_B.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
    uncover_img = uncover_img.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
    uncover_img_generate = uncover_img_generate.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)

    showResult = torch.cat(
        [originalFramesB, originalFramesContainer, generated, uncover_img_B, uncover_img, uncover_img_generate ], 0)

    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    vutils.save_image(showResult, resultImgName, nrow=this_batch_size, padding=1, normalize=False)

# print the training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log into files
    if not opt.debug:
        # write the log into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
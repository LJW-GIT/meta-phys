from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import cv2
import numpy as np
import random
import math
# from torch.utils.data import DataLoader
from dataloader import MHDataLoader
from torchvision import transforms
import scipy.io as sio

from PhysNet_META import PhysNet_padding_ED_peak

from LoadVideotrain import VIPL_train, Normaliztion, ToTensor, RandomHorizontalFlip

from TorchLossComputer import TorchLossComputer

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

# Dataset root
#VIPL_root_list = '/scratch/project_2003204/VIPL_frames_Matlab/'#没用
# VIPL_root_list = '/wrk/yuzitong/DONOTREMOVE/VIPL_frames_MTCNN_align2/'


device_ids = [0, 1]
k = 5  # 10-fold cross-validation

frames = 160  # frames = 128


# feature  -->   [ batch, channel, temporal, height, width ]
# torch.Size([1, 64, 64, 60, 60])
def FeatureMap2Heatmap(x, feature1, feature2):
    ## initial images
    ## initial images
    org_img = x[0, :, 32, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log + '/' + args.log + '_x_visual.jpg', org_img)

    ## first feature
    feature_first_frame = feature1[0, :, 32, :, :].cpu()  ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                   feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    heatmap = np.asarray(heatmap, dtype=np.uint8)

    heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # COLORMAP_WINTER     COLORMAP_JET
    heat_img = cv2.resize(heat_img, (128, 128))
    cv2.imwrite(args.log + '/' + args.log + '_x_heatmap1.jpg', heat_img)

    ## second feature
    feature_first_frame = feature2[0, :, 32, :, :].cpu()  ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                   feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()

    heatmap = np.asarray(heatmap, dtype=np.uint8)

    heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # COLORMAP_WINTER     COLORMAP_JET
    heat_img = cv2.resize(heat_img, (128, 128))
    cv2.imwrite(args.log + '/' + args.log + '_x_heatmap2.jpg', heat_img)


class Neg_Pearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds, labels):  # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

            # if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #    loss += 1 - pearson
            # else:
            #    loss += 1 - torch.abs(pearson)

            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# main function
def train(condition):
    # GPU  & log file  -->   if use DataParallel, please comment this command
    # os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log + '/' + args.log + '_con_' + str(condition) + '_log3.txt', 'w')

    # k-fold cross-validation
    for ik in range(0, 1):
        # for ik in range(7, 10):

        index = ik + 1

        print("cross-validastion: ", index)

        # VIPL_train_list = '/wrk/yuzitong/DONOTREMOVE/VIPL_list/VIPL_ECCV_train'+'%d' % (index)+'.txt'
        # VIPL_train_list = '/users/yuzitong/Protocols/rPPG/VIPL-HR/VIPL_list/VIPL_ECCV_train1.txt'

        log_file.write('cross-valid : %d' % (index))
        log_file.write("\n")
        log_file.flush()

        finetune = args.finetune
        if finetune == True:
            print('finetune!\n')
            log_file.write('finetune!\n')
            log_file.flush()

            model = PhysNet_padding_ED_peak()
            # model = PhysNet_peak_CD_old(theta = 1.0)
            
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model = model.cuda()
            model.load_state_dict(torch.load('VIPL_PhysNet160_ECCV_rPPG_fold_best/VIPL_PhysNet160_ECCV_rPPG_fold_1_20.pkl'))
            # model.load_state_dict(torch.load('PhysNet_padding_ED_peak_finetune_001Pearson_fre/PhysNet_padding_ED_peak_finetune_001Pearson_fre_1_16.pkl'))
            # CDC
            # model.load_state_dict(torch.load('VIPL_PhysNet160_CD_old01_ECCV_rPPG/VIPL_PhysNet160_CD_old01_ECCV_rPPG_1_32.pkl'))

            lr = args.lr
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)



        else:

            print('train from scratch!\n')
            log_file.write('train from scratch!\n')
            log_file.flush()

            model = PhysNet_padding_ED_peak()
            
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            # model = PhysNet_peak_CD_old(theta = 0.2)

            model = model.cuda()

            lr = args.lr
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        criterion_Binary = nn.BCELoss()  # binary segmentation

        criterion_reg = nn.MSELoss()
        criterion_L1loss = nn.L1Loss()
        criterion_class = nn.CrossEntropyLoss()
        criterion_Pearson = Neg_Pearson()

        weight_HR_reg = 0.5
        weight_ECG = 10
        echo_batches = args.echo_batches
        # scale = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
        scale = args.scale
        fold_val_reg_loss = 10000

        # train
        for epoch in range(args.epochs):  # loop over the dataset multiple times
            scheduler.step()
            if (epoch + 1) % args.step_size == 0:
                lr *= args.gamma

            loss_rPPG_avg = AvgrageMeter()
            loss_peak_avg = AvgrageMeter()
            loss_hr_rmse = AvgrageMeter()

            model.train()

            VIPL_trainDL = VIPL_train(scale,  test=False, transform=transforms.Compose(
                [Normaliztion(), RandomHorizontalFlip(), ToTensor()]))
            # dataloader_train = DataLoader(VIPL_trainDL, batch_size=1, shuffle=True, num_workers=4)  # batchsize = 4
            dataloader_train = MHDataLoader(
                args,
                VIPL_trainDL,
                batch_size=2,
                shuffle=True,
                pin_memory=not args.cpu
            )
            for i, sample_batched in enumerate(dataloader_train):
                # get the inputs
                inputs, ecg = sample_batched['video_x'].cuda(), sample_batched['ecg'].cuda()#其实这个UBFC数据集的GT是PPG信号，差的不多就不改这个ecg的命名了
                clip_average_HR, frame_rate = sample_batched['clip_average_HR'].cuda(), sample_batched['frame_rate'].cuda()
                # scale = sample_batched['scale']
                optimizer.zero_grad()

                # forward + backward + optimize
                rPPG_peak, x_visual, x_visual3232, x_visual1616 = model(inputs)
                rPPG = rPPG_peak[:, 0, :]

                # pdb.set_trace()

                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize2
                loss_rPPG = criterion_Pearson(rPPG, ecg)

                clip_average_HR = (clip_average_HR - 40)  # [40, 180]
                fre_loss = 0.0
                train_rmse = 0.0

                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                for bb in range(inputs.shape[0]):
                    fre_loss_temp, train_rmse_temp = TorchLossComputer.cross_entropy_power_spectrum_loss(rPPG[bb],clip_average_HR[bb],frame_rate[bb])
                    fre_loss = fre_loss + fre_loss_temp
                    train_rmse = train_rmse + train_rmse_temp
                fre_loss = fre_loss / inputs.shape[0]
                train_rmse = train_rmse / inputs.shape[0]

                # loss =  0.02*loss_rPPG + fre_loss
                loss = 0.1*loss_rPPG +fre_loss
                # print("loss:",loss_rPPG)
                # print("fre_loss",fre_loss)
                # loss =  loss_rPPG + fre_loss

                #原文用的是下面这个损失函数
                # loss =  loss_rPPG





                loss.backward()
                optimizer.step()

                n = inputs.size(0)
                loss_rPPG_avg.update(loss_rPPG.data, n)
                loss_peak_avg.update(fre_loss.data, n)
                loss_hr_rmse.update(train_rmse, n)

                if i % echo_batches == echo_batches - 1:  # print every 50 mini-batches

                    # visulization
                    visual = FeatureMap2Heatmap(inputs, x_visual3232, x_visual1616)

                    print('epoch:%d, mini-batch:%3d, lr=%f, NegPearson= %.4f, fre_CEloss= %.4f, hr_rmse= %.4f' % (
                    epoch + 1, i + 1, lr, loss_rPPG_avg.avg, loss_peak_avg.avg, loss_hr_rmse.avg))
                    # log written
                    log_file.write(
                        'epoch:%d, mini-batch:%3d, lr=%f, NegPearson= %.4f, fre_CEloss= %.4f, hr_rmse= %.4f' % (
                        epoch + 1, i + 1, lr, loss_rPPG_avg.avg, loss_peak_avg.avg, loss_hr_rmse.avg))
                    log_file.write("\n")
                    log_file.flush()

                    # show the ecg images

                    results_rPPG = []
                    y1 = 2 * rPPG[0].cpu().data.numpy()
                    y2 = ecg[0].cpu().data.numpy()  # +1 all positive
                    results_rPPG.append(y1)
                    results_rPPG.append(y2)
                    sio.savemat(args.log + '/' + 'rPPG.mat', {'results_rPPG': results_rPPG})

            log_file.write("\n")
            log_file.write('epoch:%d, mini-batch:%3d, lr=%f, NegPearson= %.4f, fre_CEloss= %.4f, hr_rmse= %.4f' % (
            epoch + 1, i + 1, lr, loss_rPPG_avg.avg, loss_peak_avg.avg, loss_hr_rmse.avg))
            log_file.write("\n")
            log_file.write("\n")
            log_file.flush()

            ## save model with corresponding epoch with the lowest val reg loss in one fold

            if epoch > args.epochs - 3:
                torch.save(model.state_dict(),args.log + '/' + args.log + '_con_' + str(condition) + '_%d_%d.pkl' % (index, epoch))

    print('Finished Training')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=2, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  # default=0.0001
    parser.add_argument('--step_size', type=int, default=50,
                        help='stepsize of optim.lr_scheduler.StepLR, how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 200
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="SSTTFinallog_Constrative", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--test', default=False, help='whether test')
    parser.add_argument('--version', default=3, help='version info')
    parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
    parser.add_argument('--scale', type=str, default='', help='super resolution scale')
    parser.add_argument('--cpu', action='store_true',help='use cpu only')
    args = parser.parse_args()
    if args.scale=='':
        args.scale = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
    else:
        args.scale = list(map(lambda x: float(x), args.scale.split('+')))
    #for i in [0, 1, 2, 5]:
        #train(i)
    train(args.version)

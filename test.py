from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import cv2
import numpy as np
import random
import math
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
VIPL_root_list = '/scratch/project_2003204/VIPL_frames_Matlab/'
# VIPL_root_list = '/wrk/yuzitong/DONOTREMOVE/VIPL_frames_MTCNN_align2/'


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


def test(condition, scale):
    # GPU  & log file  -->   if use DataParallel, please comment this command
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log + '/' + 'train_version' + str(condition))
    if not isExists:
        os.makedirs(args.log+ '/' + 'train_version' + str(condition))
    log_file = open(args.log + '/' + 'train_version' + str(condition) + '/'  +args.log + '_test_condition_' + 'scale'+ str(scale) + '_.txt', 'w')

    # k-fold cross-validation
    for ik in range(0, 1):
        # for ik in range(7, 10):

        index = ik + 1

        print("cross-validastion: ", index)

        log_file.write('cross-valid : %d' % (index))
        log_file.write("\n")
        log_file.flush()

        finetune = args.finetune

        print('test!\n')
        log_file.write('test!\n')
        log_file.flush()

        model = PhysNet_padding_ED_peak()
        model = torch.nn.DataParallel(model)
        model = model.cuda()

        model.load_state_dict(
            torch.load(args.log + '/' + args.log + '_con_' + str(condition) + '_%d_%d.pkl' % (1, args.epochs-1)))

        torch.no_grad()

        criterion_Pearson = Neg_Pearson()
        scale = args.scale
        loss_rPPG_avg = AvgrageMeter()
        loss_peak_avg = AvgrageMeter()
        loss_hr_rmse = AvgrageMeter()

        model.eval()
        # true_rppg_root = "data"  # ????????????
        VIPL_trainDL = VIPL_train(scale,transform=transforms.Compose([Normaliztion(), RandomHorizontalFlip(), ToTensor()]),
                                  test=True)

        dataloader_train = MHDataLoader(args,VIPL_trainDL, batch_size=1, shuffle=True, pin_memory=not args.cpu)  # batchsize = 4
        with torch.no_grad():
            for i, sample_batched in enumerate(dataloader_train):
                # get the inputs
                inputs, ecg = sample_batched['video_x'].cuda(), sample_batched['ecg'].cuda()
                clip_average_HR, frame_rate = sample_batched['clip_average_HR'].cuda(), sample_batched['frame_rate'].cuda()

                rPPG_peak, x_visual, x_visual3232, x_visual1616 = model(inputs)
                rPPG = rPPG_peak[:, 0, :]

                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize2

                for t in range(0, len(rPPG.cpu().detach().numpy().tolist())):
                    draw_scatter(ecg.cpu().detach().numpy().tolist()[t], rPPG.cpu().detach().numpy().tolist()[t], 160,
                                 str(i) + "_" + str(t), condition, scale)
                    log_file.write("-----------------------------------------------------------------------\n")
                    log_file.write("frame_rate:\n")
                    log_file.write(str(frame_rate.cpu().detach().numpy().tolist()[t]))
                    log_file.write("\n")
                    log_file.write("clip_average_HR:\n")
                    log_file.write(str(clip_average_HR.cpu().detach().numpy().tolist()[t]))
                    log_file.write("\n")
                    log_file.write("ecg:\n")
                    log_file.write(str(ecg.cpu().detach().numpy().tolist()[t]))
                    log_file.write("\n")
                    log_file.write("pre:\n")
                    log_file.write(str(rPPG.cpu().detach().numpy().tolist()[t]))
                    log_file.write("\n")
                    print("i:", i, "t:", t)

                loss_rPPG = criterion_Pearson(rPPG, ecg)

                fre_loss = 0.0
                train_rmse = 0.0

                n = inputs.size(0)
                loss_rPPG_avg.update(loss_rPPG, n)
                loss_peak_avg.update(fre_loss, n)
                loss_hr_rmse.update(train_rmse, n)

                log_file.write("\n")
                log_file.write("\n")
                log_file.flush()

    print('Finished Test')
    log_file.close()


def draw_scatter(GT, Pre, n, s, condition, scale):
    x1 = range(n)
    # ???????????????????????????R
    y1 = GT
    y2 = Pre

    # ??????????????????
    fig = plt.figure()
    # ?????????????????????1???1????????????????????????????????????
    ax1 = fig.add_subplot(1, 1, 1)
    # ????????????
    ax1.set_title('Result Analysis')
    # ?????????????????????
    ax1.set_xlabel('time')
    # ?????????????????????
    ax1.set_ylabel('rPPG')
    # ????????????
    # ax1.scatter(x1, y1, s=s, c='k', marker='.')
    # ????????????
    ax1.plot(x1, y1, c='r', ls='-')  # ?????????????????? ???
    ax1.plot(x1, y2, c='blue', ls='-')
    # ???????????????????????????
    # plt.xlim(xmax=5, xmin=0)
    # ??????
    isExists = os.path.exists("./" + args.log + "bluePre_redGT_" + str(condition) + "/" + 'scale' +str(scale))
    if not isExists:
        os.makedirs("./" + args.log + "bluePre_redGT_" + str(condition) + "/" + 'scale' +str(scale))

    plt.savefig("./" + args.log + "bluePre_redGT_" + str(condition) + "/" + 'scale' +str(scale) + '/' + s + ".png")
    plt.close()


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
    backup = args.scale 
    for scale in backup:
        args.scale = [scale]
        test(args.version, scale)


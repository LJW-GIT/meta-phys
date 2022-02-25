from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math

# clip_frames = 64
clip_frames = 160


# add three data augmentation
# 1. color, contrast
# 2. eraser, block
# 3. scale


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, sample):
        video_x, clip_average_HR, ecg_label, frame_rate, scale= sample['video_x'], sample['clip_average_HR'], sample['ecg'], sample['frame_rate'], sample['scale']
        new_video_x = (video_x - 127.5) / 128
        return {'video_x': new_video_x, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label, 'frame_rate': frame_rate, 'scale':scale}


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        video_x, clip_average_HR, ecg_label, frame_rate, scale = sample['video_x'], sample['clip_average_HR'], sample['ecg'], \
                                                          sample['frame_rate'], sample['scale']

        h, w = video_x.shape[1], video_x.shape[2]
        new_video_x = np.zeros((clip_frames, h, w, 3))

        p = random.random()
        if p < 0.5:
            # print('Flip')
            for i in range(clip_frames):
                # video
                image = video_x[i, :, :, :]
                image = cv2.flip(image, 1)
                new_video_x[i, :, :, :] = image

            return {'video_x': new_video_x, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label,
                    'frame_rate': frame_rate, 'scale':scale}
        else:
            # print('no Flip')
            return {'video_x': video_x, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label, 'frame_rate': frame_rate, 'scale':scale}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        video_x, clip_average_HR, ecg_label, frame_rate, scale = sample['video_x'], sample['clip_average_HR'], sample['ecg'], \
                                                          sample['frame_rate'], sample['scale']

        # swap color axis because
        # numpy image: (batch_size) x depth x H x W x C
        # torch image: (batch_size) x C x depth X H X W
        video_x = video_x.transpose((3, 0, 1, 2))
        video_x = np.array(video_x)

        clip_average_HR = np.array(clip_average_HR)

        frame_rate = np.array(frame_rate)

        ecg_label = np.array(ecg_label)

        scale = np.array(scale)

        return {'video_x': torch.from_numpy(video_x.astype(np.float)).float(),
                'clip_average_HR': torch.from_numpy(clip_average_HR.astype(np.float)).float(),
                'ecg': torch.from_numpy(ecg_label.astype(np.float)).float(),
                'frame_rate': torch.from_numpy(frame_rate.astype(np.float)).float(),
                'scale':torch.from_numpy(scale.astype(np.float)).float()}


# train
class VIPL_train(Dataset):
    """MAHNOB  video +  Seg_labels  """

    def __init__(self, scale, test=False, transform=None):

        if (test):
            self.vdPath_list = os.listdir("/data/maoguanhui/UBFC/")[30:42]



        else:
            self.vdPath_list = os.listdir("/data/maoguanhui/UBFC/")[:30]
            # print(self.vdPath_list)
        if ('subject20' in self.vdPath_list):
            self.vdPath_list.remove('subject20')
        self.transform = transform
        self.scale = scale

        self.frame_rate = []
        self.clhr = []
        self.Trace = [[]]
        self.idx_scale = 0
        for i in range(len(self.vdPath_list)):
            video_path = "/data/maoguanhui/UBFC/" + self.vdPath_list[i] + "/001vid.avi"
            capture = cv2.VideoCapture(video_path)
            self.frame_rate.append(capture.get(cv2.CAP_PROP_FPS))
            path = "/data/maoguanhui/UBFC/" + self.vdPath_list[i] + "/ground_truth.txt"

            f = open(path)
            data = f.readlines()  # 逐行读取txt并存成list。每行是list的一个元素，数据类型为str
            clhr_x = list(data[1].split())
            clhr_x = [str.replace('e', 'E') for str in clhr_x]
            self.clhr.append(clhr_x)


            data = list(data[0].split())
            data = [str.replace('e', 'E') for str in data]
            self.Trace.append([])
            for j in range(len(data)):
                self.Trace[i].append(float(data[j]))


    def __len__(self):
        return len(self.vdPath_list) * 7 * len(self.scale)

    def __getitem__(self, idx):
        clip = idx % 7 # 第几个剪辑片段

        # scale_idx = int( idx / (7 * len(self.vdPath_list))) #第几个 %
        # if idx % (7 * len(self.vdPath_list)) == 0:
        #     print("training_scale_idx", scale_idx)
        idx = int(idx / 7) % len(self.vdPath_list) # 第几个视频
        # print("idx",idx)
        start_frame = 160 * clip


        # 从.txt文件中读取数据


        sumHR = 0.0
        for kj in range(start_frame, start_frame + 160):
            sumHR += float(self.clhr[idx][kj])
        clip_average_HR = sumHR / 160
        if (clip_average_HR <= 40):
        #     print("clip_average_HR:", clip_average_HR, self.vdPath_list[idx], "clip", clip, "start_frame", start_frame)
             clip_average_HR = 90.0
        #     print("-------------------------------------------------")
        # print(clip_average_HR)
        # clip_average_HR = float(list(data[1].split())[0])

# /data/lijianwei/UBFC/subject1/SegFacePic
        video_x = self.get_single_video_x("/data/lijianwei/UBFC/" + self.vdPath_list[idx] + '/SegFacePic/' + "{}/".format(self.scale[self.idx_scale]), start_frame + 1)

        ecg_label = self.Trace[idx][start_frame:start_frame + 160]

        sample = {'video_x': video_x, 'frame_rate': self.frame_rate[idx], 'ecg': ecg_label, 'clip_average_HR': clip_average_HR, 'scale':self.scale[self.idx_scale]}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_video_x(self, video_jpgs_path, start_frame):
        image_name ='1' + '.png'
        image_path = os.path.join(video_jpgs_path, image_name)
        # print(image_path)
        image_shape = cv2.imread(image_path).shape
        video_x = np.zeros((clip_frames, image_shape[0], image_shape[1], 3))

        # image_id = start_frame
        for i in range(clip_frames):
            s = start_frame + i
            image_name = str(s) + '.png'

            # face video
            image_path = os.path.join(video_jpgs_path, image_name)

            tmp_image = cv2.imread(image_path)
            # cv2.imwrite('test111.jpg', tmp_image)

            if tmp_image is None:  # It seems some frames missing
                # image_id = 61
                # s = "%05d" % image_id
                # image_name = 'image_' + s + '.png'
                # image_path = os.path.join(video_jpgs_path, image_name)
                tmp_image = cv2.imread('./_1.jpg')
                print("______________________.jpg")

            # tmp_image = cv2.resize(tmp_image, (112, 112), interpolation=cv2.INTER_CUBIC)
            # tmp_image = cv2.resize(tmp_image, (96, 96), interpolation=cv2.INTER_CUBIC)

            video_x[i, :, :, :] = tmp_image

        return video_x

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

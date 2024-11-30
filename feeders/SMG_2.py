# xxx_2.py is a file used for emotion 2 classification
import os

import numpy as np
import pickle
import json
import random
import math

import torch
from tqdm import tqdm

from torch.utils.data import Dataset
from sklearn.metrics import f1_score


class Feeder(Dataset):
    def __init__(self, data_path, label_path, repeat=1, random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, point_nums=25, in_channels=3):
        # Get all training subjects
        self.point_nums = point_nums
        self.train_val_test = label_path
        self.in_channels = in_channels

        # Get samples for all validation sets
        self.data_dict = []
        for sample in os.listdir(data_path):
            self.data_dict.append(os.path.join(data_path, sample))

        self.time_steps = window_size

        self.label = []
        for sample in self.data_dict:
            label = int(os.path.splitext(sample)[0][-1])
            self.label.append(label)

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        self.data = []
        for data in tqdm(self.data_dict, desc=f'Loading {self.train_val_test} data'):
            # file_name = data['file_name']
            with open(data, 'r') as f:
                json_file = json.load(f)
            skeletons = np.array(json_file['skeletons'])
            # Get selected points for bones and faces, respectively
            sk = self.get_use_point(skeletons)
            self.data.append(sk)


    def get_use_point(self, skeletons):
        if self.point_nums == 25:
            return skeletons
        elif self.point_nums == 18:
            pass


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.data_dict) * self.repeat

    def __iter__(self):
        return self

    def rand_view_transform(self, X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        label = self.label[index % len(self.data_dict)]
        value = self.data[index % len(self.data_dict)]

        # data = self.train_process(value) if self.train_val_test == 'train' else self.test_process(value)
        data = self.data_process(value)  # TODO without augment
        data = np.transpose(data, (2, 0, 1))
        C, T, V = data.shape
        data = np.reshape(data, (C, T, V, 1))

        return data, label, index

    def top_k(self, score, top_k, video_level=False):
        rank = score.argsort()

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        clip_topk_acc = sum(hit_top_k) * 1.0 / len(hit_top_k)
        if video_level:
            video_top1 = {}
            for i, path in enumerate(self.data_dict):
                video_id = os.path.split(path)[-1].split('_')[1]
                video_top1.setdefault(f"{video_id}", []).append(hit_top_k[i])
            # print("Some valid data disappear!") if len(video_top1) != len(test_lst) else None
            # assert len(video_top1) == len(test_lst), "Some valid data disappear!"
            video_true_num = 0
            for k, v in video_top1.items():
                video_true_num = video_true_num + 1 if v.count(True) >= v.count(False) else video_true_num
            video_top1_acc = video_true_num / len(video_top1)
            return clip_topk_acc, video_top1_acc
        return clip_topk_acc

    def f1_score_video(self, score):
        pred_label = list(np.argmax(score, 1))
        video_dict_label = {}
        for i, path in enumerate(self.data_dict):
            video_name = os.path.splitext(path)[0].split('/')[-1].split('_')
            video_idx = video_name[1] + '_' + video_name[-1]
            video_dict_label.setdefault(f"{video_idx}", []).append(pred_label[i])
        video_top1 = {}
        for k, v in video_dict_label.items():
            video_pred = 1 if v.count(1) >= v.count(0) else 0
            video_top1[k] = video_pred
        video_pred_list = []
        video_true_list = []
        for k, v in video_top1.items():
            video_pred_list.append(v)
            video_true_list.append(int(k.split('_')[-1]))
        return f1_score(video_true_list, video_pred_list)

    def train_process(self, value):
        random.random()
        agx = random.randint(-60, 60)
        agy = random.randint(-60, 60)
        s = random.uniform(0.5, 1.5)

        center = value[0, 1, :]
        value = value - center
        scalerValue = self.rand_view_transform(value, agx, agy, s)

        scalerValue = np.reshape(scalerValue, (-1, 3))
        # scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0) + 1e-9)
        scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (
                np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0))
        scalerValue = scalerValue * 2 - 1
        scalerValue = np.reshape(scalerValue, (-1, self.point_nums, 3))

        data = np.zeros((self.time_steps, self.point_nums, 3))

        value = scalerValue[:, :, :]
        length = value.shape[0]

        # if length != self.time_steps:
        #     random_idx = random.sample(list(np.arange(length)) * 100, self.time_steps)
        #     random_idx.sort()
        #     data[:, :, :] = value[random_idx, :, :]

        if length != self.time_steps:
            idx = np.linspace(0, length - 1, self.time_steps).astype(np.int32)
            data[:, :, :] = value[idx, :, :]  # T,V,C
        else:
            data[...] = value[...]
        return data

    def test_process(self, value):
        random.random()
        agx = 0
        agy = 0
        s = 1.0

        center = value[0, 1, :]
        value = value - center
        scalerValue = self.rand_view_transform(value, agx, agy, s)

        scalerValue = np.reshape(scalerValue, (-1, 3))
        # scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0) + 1e-9)
        scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (
                np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0))
        scalerValue = scalerValue * 2 - 1

        scalerValue = np.reshape(scalerValue, (-1, self.point_nums, 3))

        data = np.zeros((self.time_steps, self.point_nums, 3))

        value = scalerValue[:, :, :]
        length = value.shape[0]
        if length != self.time_steps:
            idx = np.linspace(0, length - 1, self.time_steps).astype(np.int32)
            data[:, :, :] = value[idx, :, :]  # T,V,C
        else:
            data[...] = value[...]
        return data

    def data_process(self, value):
        data = np.zeros((self.time_steps, self.point_nums, self.in_channels))
        length = value.shape[0]
        if length != self.time_steps:
            idx = np.linspace(0, length - 1, self.time_steps).astype(np.int32)
            data[:, :, :] = value[idx, :, :]  # T,V,C
        else:
            data[...] = value[...]
        return data


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

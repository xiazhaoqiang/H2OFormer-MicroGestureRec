import os

import numpy as np
import json
import random
import math

from sklearn.metrics import f1_score
from tqdm import tqdm

from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, data_path, label_path, repeat=1, random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, point_nums=89, list_path=None):

        self.point_nums = point_nums
        if 'val' in label_path:
            self.train_val = 'val'

            if list_path is not None:
                with open(list_path, 'r') as lfile:
                    valid_lists = json.load(lfile)

            self.data_dict = []
            for sample in os.listdir(data_path):

                subject = sample.split(".")[0].split("_")[0] + '_' + str(int(sample.split(".")[0].split("_")[2]) - 1)
                if subject in valid_lists:
                    self.data_dict.append(os.path.join(data_path, sample))
        elif 'test' in label_path:
            self.train_val = 'test'

            if list_path is not None:
                with open(list_path, 'r') as lfile:
                    valid_lists = json.load(lfile)

            self.data_dict = []
            for sample in os.listdir(data_path):

                subject = sample.split(".")[0].split("_")[0] + '_' + str(int(sample.split(".")[0].split("_")[2]) - 1)
                if subject in valid_lists:
                    self.data_dict.append(os.path.join(data_path, sample))
        else:
            self.train_val = 'train'

            if list_path is not None:
                with open(list_path, 'r') as lfile:
                    train_lists = json.load(lfile)

            self.data_dict = []
            for sample in os.listdir(data_path):

                subject = sample.split(".")[0].split("_")[0] + '_' + str(int(sample.split(".")[0].split("_")[2]) - 1)
                if subject in train_lists:
                    self.data_dict.append(os.path.join(data_path, sample))

        self.time_steps = 52

        self.label = []
        for sample in self.data_dict:

            if not sample.split("/")[-1].startswith("0"):

                label = int(sample.split("/")[-1].split(".")[0].split("_")[0]) - 1
            else:

                if sample.split("/")[-1].split("_")[1] == "99":
                    label = 31
                else:
                    label = int(sample.split("/")[-1].split("_")[1])
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

        self.train_lst = ['0001', '0007', '0008', '0011', '0026', '0027', '0046', '0047', '0048', '0053', '0055',
                          '0057',
                          '0063', '0066', '0067', '0074', '0077', '0084', '0085', '0086', '0091', '0096', '0102',
                          '0103',
                          '0104', '0105', '0106', '0108', '0109', '0112', '0114', '0117', '0124', '0132', '0134',
                          '0140',
                          '0148', '0157', '0161', '0162', '0163', '0164', '0167', '0169', '0170', '0172', '0176',
                          '0186',
                          '0196', '0216', '0233', '0236', '0237', '0239', '0240', '0241', '0242', '0243', '0249',
                          '0252',
                          '0266', '0290', '0297', '0300', '0306', '0310', '0314', '0318', '0319', '0320', '0324',
                          '0334',
                          '0339', '0359', '0360', '0365', '0369', '0377', '0381', '0385', '0392', '0402', '0407',
                          '0410',
                          '0411', '0413', '0414', '0415', '0420', '0429', '0433', '0435', '0436', '0440', '0441',
                          '0445',
                          '0448', '0449', '0450', '0451', '0452', '0453']
        self.test_lst = ['0009', '0010', '0012', '0016', '0028', '0035', '0043', '0049', '0050', '0051', '0052', '0054',
                         '0056', '0060', '0071', '0072', '0083', '0089', '0095', '0100', '0107', '0110', '0111', '0121',
                         '0131', '0136', '0144', '0145', '0165', '0166', '0168', '0171', '0174', '0179', '0185', '0191',
                         '0193', '0212', '0220', '0223', '0230', '0231', '0232', '0234', '0235', '0238', '0244', '0246',
                         '0254', '0258', '0262', '0263', '0268', '0270', '0276', '0280', '0282', '0303', '0304', '0311',
                         '0312', '0313', '0315', '0316', '0317', '0322', '0323', '0325', '0327', '0328', '0329', '0333',
                         '0337', '0341', '0343', '0344', '0363', '0374', '0400', '0401', '0403', '0404', '0405', '0406',
                         '0408', '0416', '0417', '0419', '0421', '0428', '0430', '0431', '0437', '0438', '0439', '0442',
                         '0443', '0444', '0446', '0447']
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        self.data = []
        for data in tqdm(self.data_dict, desc=f'Loading {self.train_val} data'):
            # file_name = data['file_name']
            with open(data, 'r') as f:
                json_file = json.load(f)
            skeletons = np.array(json_file['skeletons'])
            left_hands = np.array(json_file['left_hand'])
            right_hands = np.array(json_file['right_hand'])
            face = np.array(json_file["face"])
            sk = self.get_use_point(skeletons, left_hands, right_hands, face)

            self.data.append(sk)

    def get_use_point(self, skeletons, left_hands, right_hands, face):
        if self.point_nums == 25:
            return skeletons
        elif self.point_nums == 67:
            return np.concatenate([skeletons, left_hands, right_hands], axis=1)
        elif self.point_nums == 137:
            return np.concatenate([skeletons, left_hands, right_hands, face], axis=1)
        elif self.point_nums == 89:
            return np.concatenate(
                [skeletons[:, 0:9, :], skeletons[:, 15:19, :], left_hands, right_hands, face[:, 36:, :]], axis=1)
        elif self.point_nums == 55:
            return np.concatenate(
                [skeletons[:, 0:9, :], skeletons[:, 15:19, :], left_hands, right_hands], axis=1)
        elif self.point_nums == 58:
            return np.concatenate(
                [skeletons[:, 0:8, :], skeletons[:, 15:19, :], left_hands[:, 0::4, :], right_hands[:, 0::4, :],
                 face[:, 36:, :]], axis=1)
        elif self.point_nums == 24:
            return np.concatenate(
                [skeletons[:, 0:8, :], skeletons[:, 15:19, :], left_hands[:, 0::4, :], right_hands[:, 0::4, :]], axis=1)
        elif self.point_nums == 22:
            return np.concatenate(
                [skeletons[:, 0:8, :], skeletons[:, 15:19, :], left_hands[:, 4::4, :], right_hands[:, 4::4, :]], axis=1)


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

        # data = self.train_process(value) if self.train_val == 'train' else self.test_process(value)
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
                video_id = os.path.split(path)[-1].split('_')[0]
                if video_id in self.test_lst:
                    video_top1.setdefault(f"{video_id}", []).append(hit_top_k[i])
            print("Some valid data disappear!") if len(video_top1) != len(self.test_lst) else None
            # assert len(video_top1) == len(self.test_lst), "Some valid data disappear!"
            video_true_num = 0
            for k, v in video_top1.items():
                video_true_num = video_true_num + 1 if v.count(True) >= v.count(False) else video_true_num
            video_top1_acc = video_true_num / 100
            # video_top1_acc = video_true_num / len(video_top1)
            return clip_topk_acc, video_top1_acc
        return clip_topk_acc

    def f1_score_video(self, score):
        pred_label = list(np.argmax(score, 1))
        video_dict_label = {}
        for i, path in enumerate(self.data_dict):
            video_name = os.path.split(path)[-1].split('_')
            video_idx = video_name[0] + '_' + video_name[1]
            if video_name[0] in self.test_lst:
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

        if length != self.time_steps:
            random_idx = random.sample(list(np.arange(length)) * 100, self.time_steps)
            random_idx.sort()
            data[:, :, :] = value[random_idx, :, :]
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

        idx = np.linspace(0, length - 1, self.time_steps).astype(np.int32)
        data[:, :, :] = value[idx, :, :]  # T,V,C
        return data

    def data_process(self, value):
        data = np.zeros((self.time_steps, self.point_nums, 3))
        length = value.shape[0]
        idx = np.linspace(0, length - 1, self.time_steps).astype(np.int32)
        data[:, :, :] = value[idx, :, :]  # T,V,C
        return data


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

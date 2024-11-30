#!/usr/bin/env python
from __future__ import print_function
import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm
from torchlight import DictAction
from torch import linalg as LA



def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    # training speed is too slow if set to True
    torch.backends.cudnn.deterministic = True

    # on cuda 11 cudnn8, the default algorithm is very slow
    # unlike on cuda 10, the default works well
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_mmd_loss(z, z_prior, y, num_cls):
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y == i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    l2_z_mean = LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean, z_mean[y_valid]


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--result-dir',
        default='./result_dir/SMG/SSI_SMG_e10d4',
        help='the work folder for storing results')
    parser.add_argument(
        '--weights-dir',
        default='./work_dir/SMG/SSI_SMG_e10d4',
        help='The path of the pretrain weight')
    parser.add_argument(
        '--weights-epoch',
        default=2,
        help='The path of the pretrain weight')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()
        self.load_optimizer()
        self.load_data()

        self.lr = self.arg.base_lr
        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list and len(self.arg.device) > 1:
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.arg.device,
                output_device=self.output_device)
            self.num_class = self.model.module.num_class
        else:
            self.num_class = self.model.num_class

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader_test = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.result_dir)
        print(Model)
        self.model = Model(**self.arg.model_args,
                           T=self.arg.train_feeder_args['window_size'],
                           decoder=self.arg.decoder,
                           arg=self.arg)
        print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        if self.arg.decoder:
            if self.arg.loss_rec == 'sum_mse':
                from model.loss_func import frame_sum_mse
                self.loss_rec = lambda x, y: frame_sum_mse(x, y)
            elif self.arg.loss_rec == 'mean_mse':
                from model.loss_func import frame_mean_mse
                self.loss_rec = lambda x, y: frame_mean_mse(x, y)
            elif self.arg.loss_rec == 'batch_mse':
                from model.loss_func import batch_sum_mse
                self.loss_rec = lambda x, y: batch_sum_mse(x, y)
            elif self.arg.loss_rec == 'sce':
                from model.loss_func import sce_loss
                self.loss_rec = lambda x, y: sce_loss(x, y)
            elif self.arg.loss_rec == 'sum_sce':
                from model.loss_func import frame_sum_sce
                self.loss_rec = lambda x, y: frame_sum_sce(x, y)
            elif self.arg.loss_rec == 'sum_heat':
                from model.loss_func import frame_sum_heatmse
                self.loss_rec = lambda x, y: frame_sum_heatmse(x, y, num_point=self.arg.model_args['num_point'])

        if self.arg.weights_dir:
            for filename in os.listdir(self.arg.weights_dir):
                if filename.endswith(('.pt', '.pkl')) and str(self.arg.weights_epoch) == filename.split('-')[1]:
                    self.arg.weights = os.path.join(self.arg.weights_dir, filename)
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=self.arg.momentum,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'NAdam':
            self.optimizer = optim.NAdam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.result_dir):
            os.makedirs(self.arg.result_dir)
        with open('{}/config.yaml'.format(self.arg.result_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam' or self.arg.optimizer == 'NAdam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/a_log.txt'.format(self.arg.result_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def eval(self, wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Mask Ratio: {}'.format(self.model.get_mask_ratio()))

        loss_value = []
        loss_value2 = []
        score_frag = []
        label_list = []
        pred_list = []

        step = 0
        process = tqdm(self.data_loader_test, ncols=40)
        for batch_idx, (data, label, index) in enumerate(process):
            label_list.append(label)
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                if self.arg.decoder is not None:
                    output, rec, ori = self.model(data, F.one_hot(label, num_classes=self.num_class))
                    loss_rec = self.loss_rec(rec, ori)
                else:
                    loss_rec = 0
                    output = self.model(data, F.one_hot(label, num_classes=self.num_class))
                loss = self.loss(output, label)

                score_frag.append(output.data.cpu().numpy())
                loss_value.append(loss.data.item())
                if self.arg.decoder is not None:
                    loss_value2.append(loss_rec.data.item())
                else:
                    loss_value2.append(loss_rec)

                _, predict_label = torch.max(output.data, 1)
                pred_list.append(predict_label.data.cpu().numpy())

                step += 1

            if wrong_file is not None or result_file is not None:
                predict = list(predict_label.cpu().numpy())
                true = list(label.data.cpu().numpy())
                for i, x in enumerate(predict):
                    if result_file is not None:
                        f_r.write(str(x) + ',' + str(true[i]) + '\n')
                    if x != true[i] and wrong_file is not None:
                        f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

        score = np.concatenate(score_frag)
        loss = np.mean(loss_value)
        loss_rec = np.mean(loss_value2)

        if 'SMG' in self.arg.feeder or 'iMiGUE' in self.arg.feeder:
            self.data_loader_test.dataset.sample_name = np.arange(len(score))
        if '_2' in self.arg.feeder:
            accuracy, video_acc = self.data_loader_test.dataset.top_k(score, 1, video_level=True)
            video_f1_score = self.data_loader_test.dataset.f1_score_video(score)
        else:
            accuracy = self.data_loader_test.dataset.top_k(score, 1)
            video_acc = 0
            video_f1_score = 0

        self.print_log('\tMean {} loss of {} batches: loss: {:.4f}. loss_rec: {:.4f}.'.format(
            'test', len(self.data_loader_test), loss, loss_rec))
        if '_2' in self.arg.feeder:
            self.print_log('\tClip-Top1-Acc: {:.2f}%'.format(100 * accuracy))
            self.print_log('\tVideo-Top1-Acc: {:.2f}%'.format(100 * video_acc))
            self.print_log('\tVideo-F1-score: {:.4f}'.format(video_f1_score))
        else:
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader_test.dataset.top_k(score, k)))

        # acc for each class:
        label_list = np.concatenate(label_list)
        pred_list = np.concatenate(pred_list)
        confusion = confusion_matrix(label_list, pred_list)
        list_diag = np.diag(confusion)
        list_raw_sum = np.sum(confusion, axis=1)
        each_acc = list_diag / list_raw_sum
        with open('{}/each_class_acc.csv'.format(self.arg.result_dir, 'test'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(each_acc)
            writer.writerows(confusion)
        return format(100 * accuracy, '.2f'), format(100 * video_acc, '.2f')

    def start(self):
        wf = f'{self.arg.result_dir}/wrong.txt'
        rf = f'{self.arg.result_dir}/right.txt'
        self.arg.print_log = False
        clip_acc, video_acc = self.eval(wrong_file=wf, result_file=rf)
        self.arg.print_log = True

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print_log(f'Clip Accuracy: {clip_acc}%')
        self.print_log(f'Video Accuracy: {video_acc}%')
        self.print_log(f'Result dir: {self.arg.result_dir}')
        self.print_log(f'Model total number of params: {num_params}')
        self.print_log(f'Weight decay: {self.arg.weight_decay}')
        self.print_log(f'Base LR: {self.arg.base_lr}')
        self.print_log(f'Batch Size: {self.arg.batch_size}')
        self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
        self.print_log(f'seed: {self.arg.seed}')


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.weights_dir is not None:
        with open(p.weights_dir + '/config.yaml', 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                # add new keys to arg
                parser.add_argument('--' + k, default=default_arg[k])
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()

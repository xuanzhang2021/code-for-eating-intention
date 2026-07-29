#!/usr/bin/env python
# pylint: disable=W0201
from collections import Counter
from pyexpat import model
import sys
import argparse
import yaml
from feeder.feeder import Feeder
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
# from torchlight import str2bool
# from torchlight import DictAction
# from torchlight import import_class

from torchlight.torchlight.io import str2bool
from torchlight.torchlight.io import DictAction
from torchlight.torchlight.io import import_class
from torch.utils.data import DataLoader, random_split

from .io import IO
import random
import matplotlib.pyplot as plt
import os



class Processor(IO):
    """
        Base Processor
    """
    def __init__(self, argv=None):
        random.seed(1)
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()
        self.count_labels_by_subset

        self.training_loss_log = []  # 用于记录训练损失
        self.validation_loss_log = []  # 用于记录验证损失
        self.validation_acc_log = []

    def init_environment(self):

        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def load_optimizer(self):
        pass

    def save_loss_logs(self):
        # Convert lists to numpy arrays
        training_loss_array = np.array(self.training_loss_log)
        validation_loss_array = np.array(self.validation_loss_log)
        np.savetxt('training_loss_log.txt', training_loss_array)
        np.savetxt('validation_loss_log.txt', validation_loss_array)
        print("Loss logs saved to 'training_loss_log.txt' and 'validation_loss_log.txt'")

    def plot_loss_curve(self, training_loss, validation_loss, validation_acc):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss', color='blue', marker='o')
        plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Loss', color='orange',
                 marker='x')
        plt.plot(range(1, len(validation_acc) + 1), validation_acc, label='Validation acc', color='red',
                 marker='x')

        plt.title('Loss Convergence Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig('loss_convergence_curve.png')
        plt.show()


    def count_labels_by_subset(self, full_dataset, subset):
    # subset 是 random_split 得到的 Subset
        c = Counter()
        for idx in subset.indices:
            y = full_dataset[idx][1]  # (x, y) 的第2个是 label
            # y 可能是 tensor
            if torch.is_tensor(y):
                y = int(y.item())
            c[y] += 1
        return c
    
    def load_data(self):
        """
        加载完整数据集后，自动划分 70% / 20% / 10% 数据为 train / valid / test。
        """

        Feeder = import_class(self.arg.feeder)

        # === 1. 获取 Feeder 参数 ===
        if hasattr(self.arg, "feeder_args"):
            feeder_args = self.arg.feeder_args
        elif hasattr(self.arg, "train_feeder_args"):
            feeder_args = self.arg.train_feeder_args  # 兼容旧版配置
        else:
            raise ValueError("请在 YAML 中提供 feeder_args")

        if "debug" not in feeder_args:
            feeder_args["debug"] = self.arg.debug

        # === 2. 读取完整数据集 ===
        full_dataset = Feeder(**feeder_args)
        dataset_len = len(full_dataset)
        print(f"✅ Loaded full dataset: {dataset_len} samples")
        # print("full_dataset_sample:", full_dataset[0])

        # === 3. 按比例划分 ===
        train_ratio, val_ratio = 0.7, 0.2
        train_size = int(train_ratio * dataset_len)
        val_size = int(val_ratio * dataset_len)
        test_size = dataset_len - train_size - val_size

        train_set, val_set, test_set = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"✅ Split dataset: train={train_size}, valid={val_size}, test={test_size}")

        train_counts = self.count_labels_by_subset(full_dataset, train_set)
        val_counts   = self.count_labels_by_subset(full_dataset, val_set)
        test_counts  = self.count_labels_by_subset(full_dataset, test_set)

        print("📊 Train label counts:", dict(train_counts))
        print("📊 Valid label counts:", dict(val_counts))
        print("📊 Test  label counts:", dict(test_counts))

        # === 4. 构建 DataLoader ===
        ngpu = torchlight.torchlight.ngpu(self.arg.device)

        self.data_loader = {
            "train": DataLoader(
                train_set,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * ngpu,
                drop_last=True,
            ),
            "valid": DataLoader(
                val_set,
                batch_size=self.arg.valid_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * ngpu,
            ),
            "test": DataLoader(
                test_set,
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * ngpu,
            ),
        }

        print("✅ DataLoader built successfully (train/valid/test).")

    '''
    def load_data(self):
        """
        分别加载预先划分好的 train / valid / test 数据集
        """
        Feeder = import_class(self.arg.feeder)
        ngpu = torchlight.torchlight.ngpu(self.arg.device)

        # 加载训练集
        train_feeder_args = self.arg.train_feeder_args.copy()
        train_feeder_args["debug"] = self.arg.debug
        train_set = Feeder(**train_feeder_args)
    
        # 加载验证集
        valid_feeder_args = self.arg.valid_feeder_args.copy()
        valid_feeder_args["debug"] = self.arg.debug
        valid_set = Feeder(**valid_feeder_args)
    
        # 加载测试集
        test_feeder_args = self.arg.test_feeder_args.copy()
        test_feeder_args["debug"] = self.arg.debug
        test_set = Feeder(**test_feeder_args)

        print(f"✅ Loaded datasets: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}")

        # 构建 DataLoader
        self.data_loader = {
            "train": DataLoader(
                train_set,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * ngpu,
                drop_last=True,
            ),
            "valid": DataLoader(
                valid_set,
                batch_size=self.arg.valid_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * ngpu,
            ),
            "test": DataLoader(
                test_set,
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * ngpu,
            ),
        }

        print("✅ DataLoader built successfully (train/valid/test).")
    '''

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()


    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        mae = float('inf')
        best_val_loss = float('inf')
        best_model_path = None
        # training phase
        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train()
                self.io.print_log('Done.')

                # save model
                if ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))

                    val_loss, val_acc = self.test()     
                    self.show_epoch_info() 
                    if val_loss < best_val_loss:  # 保存验证集上表现最好的模型
                        best_val_loss = val_loss
                        filename = f'epoch{epoch+1}_model_best_valloss_{best_val_loss:.6f}.pt'
                        self.io.save_model(self.model, filename)
                        best_model_path = os.path.join(self.arg.work_dir, filename)
                    self.io.print_log('Done.')
        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test()      # 这块的test，实际调用的是recognition.py中的test函数
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                # result_dict = dict(zip(self.data_loader['test'].dataset.sample_name, self.result))
                # print('111111', self.result.shape)

                # 确保测试数据和结果长度一致
                if len(self.data_loader['test'].dataset) != len(self.result):
                    raise ValueError("The number of test samples and results do not match!")

                # 获取测试数据集和预测结果的对应关系
                dataset_values = [self.data_loader['test'].dataset[i] for i in range(len(self.data_loader['test'].dataset))]
                print(self.data_loader['test'].dataset[1][1])

                # 构建结果字典，包含数据集值和预测结果
                result_dict = {
                    i: {'data': dataset_values[i][1], 'prediction': self.result[i]} for i in range(len(self.result))}

                # 保存到文件
                self.io.save_pkl(result_dict, 'test_result.pkl')
                print("Results and dataset values saved to 'test_result.pkl'")

            # print(result_dict)

        # 所有 epoch 完成后，训练结束后：用 best 模型在 test 集上评估一次 ===
        if best_model_path is not None:
            self.io.print_log(f'Load best model for final test: {best_model_path}')
            self.io.load_weights(self.model, best_model_path)  # 依据你 IO 里的接口，可能叫 load_model / load_weights
            test_loss, test_acc = self.test(split='test')
            self.io.print_log(f'Final Test (best on valid) | loss={test_loss:.6f} acc={test_acc:.4f}')
        else:
            self.io.print_log('[WARN] best_model_path is None, skip final test.')

        # 所有 epoch 完成后保存损失日志和绘制曲线
        self.save_loss_logs()
        training_loss = self.training_loss_log
        validation_loss = self.validation_loss_log
        print('training_loss:', training_loss)
        print('validation_loss:', validation_loss)
        print('validation_acc:', self.validation_acc_log)
        self.plot_loss_curve(training_loss, validation_loss, self.validation_acc_log)    
        print(f"Learned sim_beta: {torch.sigmoid(self.model.sim_beta).item():.4f}")
        print(f"Learned sim_sigma: {self.model.sim_sigma.item():.4f}")
 


    @staticmethod
    def get_parser(add_help=False):

        parser = argparse.ArgumentParser( add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=True, help='if True, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=5, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=20, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=1, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--valid_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for valid')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--valid_batch_size', type=int, default=256, help='valid batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        # weights: train的时候用None, test的时候写具体的路径
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        # parser.add_argument('--weights', default='/home/zhangxuan/sourcecode/EVA-GCN-main/work_dir_19_points_FAN_6_layers_train_on_300W_LP_test_on_BIWI/GCN_head_pose/epoch107_model_mae_4.076589107513428.pt/epoch400_model.pt', help='the weights for network initialization')
        # parser.add_argument('--weights', default='/home/zhangxuan/sourcecode/EVA-GCN-main/work_dir_19_points_FAN_6_layers_train_on_300W_LP_test_on_BIWI/GCN_head_pose/epoch107_model_mae_4.076589107513428.pt/epoch299_model.pt', help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        # endregion yapf: enable

        return parser

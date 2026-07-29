#!/usr/bin/env python
# pylint: disable=W0201
from posixpath import split
import sys
import os
import argparse
from unittest import loader
import yaml
import numpy as np
import time
# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torch.autograd import Variable
# from torchlight import str2bool
# from torchlight import DictAction
# from torchlight import import_class
from torchlight.torchlight.io import str2bool
from torchlight.torchlight.io import DictAction
from torchlight.torchlight.io import import_class
# import numpy as np

from .processor import Processor
import random


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        random.seed(1)
        self.model.apply(weights_init)
        # self.loss = nn.CrossEntropyLoss()    # 适用于多分类的任务
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)    # label_smoothing=0.0：94.8，0.1： 94.3，0.2：93.04  0.5：90.2
        # self.loss = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 二分类
        # self.r_loss = nn.MSELoss()
        # self.c_loss = nn.CrossEntropyLoss()
        # self.mae = nn.L1Loss()


    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.2**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
          
        else:
            self.lr = self.arg.base_lr
        print("epoch =", self.meta_info['epoch'], "lr =", self.optimizer.param_groups[0]['lr'])
    
    def train(self):
    # train mode
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']

    # === 统计量（按样本数加权）===
        loss_sum, n = 0.0, 0
        correct = 0

        # for i, (data, label) in enumerate(loader):
        for i, batch in enumerate(loader):
            if len(batch) == 2:
                data, label = batch
                idx = None
            else:
                data, label, idx = batch[:3]

            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

        # forward
            # print(data.shape)
            output = self.model(data)                 # [N, C]
            loss = self.loss(output, label)           # CE over logits

        # backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        # === batch 级统计 ===
            bs = label.size(0)
            loss_sum += loss.item() * bs
            n += bs

            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()

        # iter info（用于 show_iter_info）
            self.iter_info['loss'] = loss.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
        # 如需打印每隔 log_interval 的 iter 日志，取消注释：
        # self.show_iter_info()

            self.meta_info['iter'] += 1

    # === epoch 级统计 ===
        mean_loss = loss_sum / n if n > 0 else 0.0
        train_acc = correct / n if n > 0 else 0.0

        self.epoch_info['mean loss in training'] = mean_loss
        self.epoch_info['train Accuracy'] = train_acc  # 新增：训练集准确率

        self.training_loss_log.append(mean_loss)
        self.show_epoch_info()
        self.io.print_timer()

    # def test(self, evaluation=True):
    # def test(self, evaluation=True, split=None):
    def test(self, evaluation=True, split=None, save_attention=False):
        self.model.eval()

        if split is None:
            split = 'valid' if self.arg.phase == 'train' else 'test'
        loader = self.data_loader[split]
        print("model.training =", self.model.training)  # 必须输出 False

        num_classes = self.arg.model_args['num_class']
        conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

        loss_sum, n = 0.0, 0
        correct = 0

        result_frag = []
        label_frag = []

        # ===== 新增：attention 收集容器 =====
        all_attn = []
        all_S_attn = []
        all_A_eff = []
        all_idx = []
        # ===== 新增结束 =====

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if len(batch) == 2:
                    data, label = batch
                    idx = None
                else:
                    data, label, idx = batch[:3]

                data = data.float().to(self.dev)
                label = label.long().to(self.dev)

                # ===== 输入检查 =====
                if not torch.isfinite(data).all():
                    self.io.print_log("\t[WARN] non-finite data detected in batch, skip forward")
                    continue
                if not torch.isfinite(label).all():
                    self.io.print_log("\t[WARN] non-finite label detected in batch")
                    continue
                # ===== 输入检查结束 =====

                # ===== 关键修改：支持提取 attention =====
                if save_attention:
                    output, attn_dict = self.model(data, return_attn=True)

                    if 'alpha_x' in attn_dict and attn_dict['alpha_x'] is not None:
                        all_attn.append(attn_dict['alpha_x'].detach().cpu())

                    if 'S_attn' in attn_dict and attn_dict['S_attn'] is not None:
                        all_S_attn.append(attn_dict['S_attn'].detach().cpu())

                    if 'A_eff' in attn_dict and attn_dict['A_eff'] is not None:
                        all_A_eff.append(attn_dict['A_eff'].detach().cpu())

                    if idx is not None:
                        if torch.is_tensor(idx):
                            all_idx.append(idx.detach().cpu())
                        else:
                            all_idx.append(torch.tensor(idx))
                else:
                    output = self.model(data)
                # ===== 关键修改结束 =====

                loss = self.loss(output, label)

                # ===== spike 样本定位 =====
                log_probs = output.log_softmax(dim=1)
                nll = -log_probs.gather(1, label.view(-1, 1)).squeeze(1)

                batch_max = float(nll.max().item())
                batch_mean = float(nll.mean().item())

                spike_mask = nll > 10
                if spike_mask.any():
                    spike_pos = spike_mask.nonzero(as_tuple=False).view(-1)[:10]
                    for j in spike_pos:
                        j = int(j.item())
                        true_y = int(label[j].item())
                        pred_y = int(output[j].argmax().item())

                        probs1 = output[j].softmax(dim=0)
                        p_true = float(probs1[true_y].item())
                        p_pred = float(probs1[pred_y].item())
                        nll_j = float(nll[j].item())

                        if idx is not None:
                            idx_j = int(idx[j].item()) if torch.is_tensor(idx) else int(idx[j])
                        else:
                            idx_j = -1

                        self.io.print_log(
                            f"\t[WARN] spike sample: idx={idx_j}, nll={nll_j:.4f}, "
                            f"true={true_y}, pred={pred_y}, p_true={p_true:.6e}, p_pred={p_pred:.6e}"
                        )

                    self.io.print_log(
                        f"\t[WARN] spike batch summary: mean_nll={batch_mean:.4f}, max_nll={batch_max:.4f}"
                    )
                # ===== spike 定位结束 =====

                bs = label.size(0)
                loss_sum += loss.item() * bs
                n += bs

                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()

                # ===== 混淆矩阵 =====
                true_cpu = label.detach().view(-1).to('cpu')
                pred_cpu = pred.detach().view(-1).to('cpu')
                idx_cm = true_cpu * num_classes + pred_cpu
                cm = torch.bincount(idx_cm, minlength=num_classes * num_classes)
                conf_mat += cm.view(num_classes, num_classes)
                # ===== 混淆矩阵结束 =====

                probs = output.softmax(dim=1)
                result_frag.append(probs.cpu().numpy())
                label_frag.append(label.cpu().numpy())

        # ===== 防止空 loader / 全部跳过 =====
        if n == 0:
            self.io.print_log(f"[WARN] No valid samples found in split={split}.")
            avg_validation_loss = 0.0
            mean_accuracy = 0.0
            self.result = np.array([])
            if evaluation:
                self.label = np.array([])
            return avg_validation_loss, mean_accuracy

        avg_validation_loss = loss_sum / n
        mean_accuracy = correct / n

        # ===== 由混淆矩阵计算额外指标 =====
        support = conf_mat.sum(dim=1)
        tp = conf_mat.diag()

        recall = tp.float() / support.clamp(min=1).float()
        pred_count = conf_mat.sum(dim=0)
        precision = tp.float() / pred_count.clamp(min=1).float()
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

        balanced_acc = recall.mean().item()
        macro_f1 = f1.mean().item()

        epoch = int(self.meta_info.get('epoch', 0))
        if ((epoch + 1) % 50 == 0) or (epoch == 0) or (epoch + 1 == self.arg.num_epoch):
            self.io.print_log(
                f"Eval split={split} | "
                f"loss={avg_validation_loss:.6f} acc={mean_accuracy:.4f} "
                f"balanced_acc={balanced_acc:.4f} macro_f1={macro_f1:.4f}"
            )
            self.io.print_log(f"Support per class: {support.tolist()}")
            self.io.print_log(f"Precision per class: {[round(p.item(), 4) for p in precision]}")
            self.io.print_log(f"Recall per class: {[round(r.item(), 4) for r in recall]}")
            self.io.print_log(f"F1-score per class: {[round(f.item(), 4) for f in f1]}")
        # ===== 额外指标结束 =====

        self.validation_loss_log.append(avg_validation_loss)
        self.validation_acc_log.append(mean_accuracy)
        self.epoch_info['validation_loss'] = avg_validation_loss
        self.epoch_info[f'{split} Accuracy'] = mean_accuracy

        self.result = np.concatenate(result_frag, axis=0)
        if evaluation:
            self.label = np.concatenate(label_frag, axis=0)

        # ===== 保存测试分类结果 =====
        if split == 'test':
            pred_label = self.result.argmax(axis=1)
            np.savez( 
                os.path.join('test_results.npz'),
                y_true=self.label,
                y_pred=pred_label,
                y_prob=self.result
            )
            self.io.print_log("✅ Test results saved to test_results.npz")
        # ===== 保存测试分类结果结束 =====

        # ===== 新增：保存 attention =====
        if save_attention:
            if len(all_attn) > 0:
                alpha_x_all = torch.cat(all_attn, dim=0).numpy()
                np.save(os.path.join(self.arg.work_dir, f'{split}_alpha_x.npy'), alpha_x_all)
                self.io.print_log(
                    f"✅ Attention saved: {split}_alpha_x.npy | shape={alpha_x_all.shape}"
                )
            else:
                self.io.print_log(f"[WARN] No alpha_x collected for split={split}")

            # S_attn / A_eff 通常不是逐样本变化很大的量，保存第一个 batch 即可
            if len(all_S_attn) > 0:
                s_attn_np = all_S_attn[0].numpy()
                np.save(os.path.join(self.arg.work_dir, f'{split}_S_attn.npy'), s_attn_np)
                self.io.print_log(
                    f"✅ S_attn saved: {split}_S_attn.npy | shape={s_attn_np.shape}"
                )

            if len(all_A_eff) > 0:
                a_eff_np = all_A_eff[0].numpy()
                np.save(os.path.join(self.arg.work_dir, f'{split}_A_eff.npy'), a_eff_np)
                self.io.print_log(
                    f"✅ A_eff saved: {split}_A_eff.npy | shape={a_eff_np.shape}"
                )

            if evaluation:
                np.save(os.path.join(self.arg.work_dir, f'{split}_labels.npy'), self.label)
                self.io.print_log(
                    f"✅ Labels saved: {split}_labels.npy | shape={self.label.shape}"
                )

            if len(all_idx) > 0:
                idx_all = torch.cat(all_idx, dim=0).numpy()
                np.save(os.path.join(self.arg.work_dir, f'{split}_sample_idx.npy'), idx_all)
                self.io.print_log(
                    f"✅ Sample indices saved: {split}_sample_idx.npy | shape={idx_all.shape}"
                )
        # ===== 保存 attention 结束 =====

        return avg_validation_loss, mean_accuracy

    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser

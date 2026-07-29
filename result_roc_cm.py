from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from itertools import cycle
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import time

import seaborn as sns
import argparse
import os
import glob
import matplotlib.patches as patches


# 类别设置 这个是journal结果用的图
N_CLASSES = 5
CLASS_NAMES = ['eating', 'refusal', 'intention', 'waiting', 'social']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 4个模型的颜色


def plot_confusion_picture(cm):
    classes = ['eating', 'refusal', 'intention', 'waiting', 'social']
    proportion = []
    plt.figure(figsize=(10, 8))  # 添加图像大小设置
    plt.rcParams['font.size'] = 24
    plt.rcParams['font.family'] = 'Times New Roman'
    length = len(cm)
    print(length)
    for i in cm:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)
    pshow = np.array(pshow).reshape(length, length)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=24)
    plt.yticks(tick_marks, classes, fontsize=24)

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (cm.size, 2))
    for i, j in iters:
        # 计算文本颜色阈值
        thresh = proportion.max() / 2.0
        text_color = 'white' if proportion[i, j] > thresh else 'black'
        
        # 合并数量和百分比为一行，完全居中显示
        text_content = f"{cm[i, j]}\n({pshow[i, j]})"
        plt.text(j, i, text_content, va='center', ha='center', fontsize=20, color=text_color)

    plt.ylabel('True Label', fontsize=24)
    plt.xlabel('Predicted Label', fontsize=24)
    plt.tight_layout()
    plt.show()


def measure_performance(y_test_scale, y_predict, y_correction):
    n = len(y_test_scale)
    mae = np.sum(np.abs(y_test_scale - y_predict)) / n
    print('prediction_mae', mae)

    correct = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_test_scale[i]:
            correct += 1
    accuracy = correct / n
    print('prediction_accuracy_yhat', accuracy)
    correct1 = 0
    for i in range(len(y_predict)):
        if y_correction[i] == y_test_scale[i]:
            correct1 += 1
    accuracy1 = correct1 / n
    print('prediction_accuracy_y_correction', accuracy1)

    return mae, accuracy, accuracy1


def compute_roc_per_class(y_true, y_prob, n_classes):
    """计算每个类别的 ROC 曲线"""
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    
    return fpr_dict, tpr_dict, roc_auc_dict


def compute_micro_roc(y_true, y_prob, n_classes):
    """计算 Micro-average ROC"""
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def compute_macro_roc(y_true, y_prob, n_classes):
    """计算 Macro-average ROC"""
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr_list, tpr_list = [], []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    # 插值到统一的 FPR 点
    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
    mean_tpr /= n_classes
    
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc

# sample level
# ========== 加载数据 ==========
# resnet-lstm
# data0 = np.load('/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/case_result/sequence_level_predictions_20260209_215208.npz')
# data0 = np.load('/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/tie_result/sequence_level_predictions_20260421_122615_resnet_lstm.npz')   # 91.23
data0 = np.load('/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/tie_result/sequence_level_predictions_20260519_220534_resnet_lstm.npz')   # 90.89
# print(data0.files)
# for key in data0.files:
    # print(f"{key}: {data0[key][:2]}")
true_data0 = data0['y_true']
pred_pro0 = data0['y_prob']
predicted_label0 = data0['y_pred']

# resnet-lstm-gaussian
data1 = np.load('/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/tie_result/sequence_level_predictions_20260421_120551_gaussian.npz')
true_data1 = data1['y_true']
pred_pro1 = data1['y_prob']
predicted_label1 = data1['y_pred']

# test_results_28lmk+pose_stgcn.npz 90.71
# test_results_28lmk+pose_stgcn1.npz 88.91
# test_results_28lmk+pose_stgcn2.npz 91.23
data2 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn.npz')
true_data2 = data2['y_true']
pred_pro2 = data2['y_prob']
predicted_label2 = data2['y_pred']

data3 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_2sagcn3.npz')
true_data3 = data3['y_true']
pred_pro3 = data3['y_prob']
predicted_label3 = data3['y_pred']

data4 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_ctrgcn.npz')
true_data4 = data4['y_true']
pred_pro4 = data4['y_prob']
predicted_label4 = data4['y_pred']

data5 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+simi+gat+filt1.npz')
true_data5 = data5['y_true']
pred_pro5 = data5['y_prob']
predicted_label5 = data5['y_pred']

# 检查 data3 的数据情况
print("=== 检查 data3 ===")
print("true_data3 类型:", type(true_data3))
print("true_data3 dtype:", true_data3.dtype)
print("true_data3 shape:", true_data3.shape)
print("true_data3 前10个值:\n", true_data3[:10])
print("true_data3 唯一值:", np.unique(true_data3))

print("\npredicted_label3 类型:", type(predicted_label3))
print("predicted_label3 dtype:", predicted_label3.dtype)
print("predicted_label3 shape:", predicted_label3.shape)
print("predicted_label3 前10个值:\n", predicted_label3[:10])
print("predicted_label3 唯一值:", np.unique(predicted_label3))


# ========== 计算混淆矩阵，并绘制热图 ==========
cm0 = confusion_matrix(true_data0, predicted_label0)
plot_confusion_picture(cm0)

cm1 = confusion_matrix(true_data1, predicted_label1)
plot_confusion_picture(cm1)

cm2 = confusion_matrix(true_data2, predicted_label2)
plot_confusion_picture(cm2)

cm3 = confusion_matrix(true_data3, predicted_label3)
plot_confusion_picture(cm3)

cm4 = confusion_matrix(true_data4, predicted_label4)
plot_confusion_picture(cm4)

cm5 = confusion_matrix(true_data5, predicted_label5)
plot_confusion_picture(cm5)

# ========== 计算 accuracy precision recall F1 ==========
print('sample level Resnet_lstm:')
print(classification_report(true_data0, predicted_label0, digits=4))
measure_performance(true_data0, predicted_label0, predicted_label0)

print('sample level Resnet_lstm_with_gaussian:')
print(classification_report(true_data1, predicted_label1, digits=4))
measure_performance(true_data1, predicted_label1, predicted_label1)

print('sample level STGCN:')
print(classification_report(true_data2, predicted_label2, digits=4))
measure_performance(true_data2, predicted_label2, predicted_label2)

print('sample level 2s-AGCN')
print(classification_report(true_data3, predicted_label3, digits=4))
measure_performance(true_data3, predicted_label3, predicted_label3)

print('sample level CTRCN:')
print(classification_report(true_data4, predicted_label4, digits=4))
measure_performance(true_data4, predicted_label4, predicted_label4)

print('sample level STGCN with improvement:')
print(classification_report(true_data5, predicted_label5, digits=4))
measure_performance(true_data5, predicted_label5, predicted_label5)

# suject level
data0 = np.load('/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/tie_result/sequence_level_predictions_20260519_220534_resnet_lstm.npz')   # 90.89
true_data0 = data0['y_true']
pred_pro0 = data0['y_prob']
predicted_label0 = data0['y_pred']

# resnet-lstm-gaussian
data1 = np.load('/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/tie_result/sequence_level_predictions_20260421_120551_gaussian.npz')
true_data1 = data1['y_true']
pred_pro1 = data1['y_prob']
predicted_label1 = data1['y_pred']

data2 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/subject-level/20260721/test_results_28lmk+pose_stgcn0.npz')
true_data2 = data2['y_true']
pred_pro2 = data2['y_prob']
predicted_label2 = data2['y_pred']

data3 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_2sagcn3.npz')
true_data3 = data3['y_true']
pred_pro3 = data3['y_prob']
predicted_label3 = data3['y_pred']

data4 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/subject-level/20260721/test_results_28lmk+pose_ctrgcn2.npz')
true_data4 = data4['y_true']
pred_pro4 = data4['y_prob']
predicted_label4 = data4['y_pred']

data5 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/subject-level/20260721/test_results_28lmk+pose_skateformer3.npz')
true_data5 = data5['y_true']
pred_pro5 = data5['y_prob']
predicted_label5 = data5['y_pred']

data6 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/subject-level/20260721/test_results_28lmk+pose_stgcn86.npz')
true_data6 = data6['y_true']
pred_pro6 = data6['y_prob']
predicted_label6 = data6['y_pred']

# ========== 计算 accuracy precision recall F1 ==========
print('subject level Resnet_lstm:')
print(classification_report(true_data0, predicted_label0, digits=4))
measure_performance(true_data0, predicted_label0, predicted_label0)

print('subject level Resnet_lstm_with_gaussian:')
print(classification_report(true_data1, predicted_label1, digits=4))
measure_performance(true_data1, predicted_label1, predicted_label1)

print('subject level STGCN:')
print(classification_report(true_data2, predicted_label2, digits=4))
measure_performance(true_data2, predicted_label2, predicted_label2)

print('subject level 2s-AGCN')
print(classification_report(true_data3, predicted_label3, digits=4))
measure_performance(true_data3, predicted_label3, predicted_label3)

print('subject level CTRCN:')
print(classification_report(true_data4, predicted_label4, digits=4))
measure_performance(true_data4, predicted_label4, predicted_label4)

print('subject level skateformer:')
print(classification_report(true_data5, predicted_label5, digits=4))
measure_performance(true_data5, predicted_label5, predicted_label5)

print('subject level STGCN8:')
print(classification_report(true_data6, predicted_label6, digits=4))
measure_performance(true_data6, predicted_label6, predicted_label6)

# ========== 准备所有模型的数据 ==========
models_data = [
    {'name': 'Resnet_lstm', 'y_true': true_data0, 'y_prob': pred_pro0, 'color': '#1f77b4'},
    {'name': 'Resnet_lstm_with_gaussian', 'y_true': true_data1, 'y_prob': pred_pro1, 'color': '#ff7f0e'},
    {'name': 'STGCN', 'y_true': true_data2, 'y_prob': pred_pro2, 'color': '#2ca02c'},
    {'name': 'STGCN with improvement', 'y_true': true_data3, 'y_prob': pred_pro3, 'color': '#d62728'},
]

# ========== 绘制每个类别的 ROC 曲线 ==========
fig, axes = plt.subplots(3, 2, figsize=(18, 12))
axes = axes.flatten()

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'

for class_idx in range(N_CLASSES):
    ax = axes[class_idx]
    
    for model in models_data:
        fpr_dict, tpr_dict, roc_auc_dict = compute_roc_per_class(
            model['y_true'], model['y_prob'], N_CLASSES
        )
        ax.plot(fpr_dict[class_idx], tpr_dict[class_idx], 
                color=model['color'], lw=2,
                label=f"{model['name']} (AUC={roc_auc_dict[class_idx]:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {CLASS_NAMES[class_idx]}', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

# 第6个子图：Micro-average ROC
ax = axes[5]
for model in models_data:
    fpr_micro, tpr_micro, auc_micro = compute_micro_roc(
        model['y_true'], model['y_prob'], N_CLASSES
    )
    ax.plot(fpr_micro, tpr_micro, color=model['color'], lw=2,
            label=f"{model['name']} (AUC={auc_micro:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Micro-average ROC Curve', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
timestamp = time.strftime("%Y%m%d_%H%M%S")
plt.savefig(f"roc_per_class_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"roc_per_class_{timestamp}.eps", format='eps', dpi=600, bbox_inches='tight')
plt.show()


# ========== 单独绘制 Micro-average 和 Macro-average ROC ==========
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Times New Roman'

# ============ Micro-average ROC ============
fig1, ax1 = plt.subplots(figsize=(8, 6))

for model in models_data:
    fpr_micro, tpr_micro, auc_micro = compute_micro_roc(
        model['y_true'], model['y_prob'], N_CLASSES
    )
    ax1.plot(fpr_micro, tpr_micro, color=model['color'], lw=2,
             label=f"{model['name']} (AUC={auc_micro:.3f})")

ax1.plot([0, 1], [0, 1], 'k--', lw=1)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate', fontsize=14)
ax1.set_ylabel('True Positive Rate', fontsize=14)
ax1.set_title('Micro-average ROC Curve', fontsize=16)
ax1.legend(loc='lower right', fontsize=12)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"roc_micro_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"roc_micro_{timestamp}.eps", format='eps', dpi=600, bbox_inches='tight')
plt.show()

# ============ Macro-average ROC ============
fig2, ax2 = plt.subplots(figsize=(8, 6))

for model in models_data:
    fpr_macro, tpr_macro, auc_macro = compute_macro_roc(
        model['y_true'], model['y_prob'], N_CLASSES
    )
    ax2.plot(fpr_macro, tpr_macro, color=model['color'], lw=2,
             label=f"{model['name']} (AUC={auc_macro:.3f})")

ax2.plot([0, 1], [0, 1], 'k--', lw=1)
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=14)
ax2.set_ylabel('True Positive Rate', fontsize=14)
ax2.set_title('Macro-average ROC Curve', fontsize=16)
ax2.legend(loc='lower right', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"roc_macro_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"roc_macro_{timestamp}.eps", format='eps', dpi=600, bbox_inches='tight')
plt.show()


# ========== 打印各模型各类别的 AUC 值汇总表 ==========
print("\n" + "="*80)
print("各模型各类别 AUC 值汇总")
print("="*80)

header = f"{'Model':<25}" + "".join([f"{name:<12}" for name in CLASS_NAMES]) + f"{'Micro':<12}{'Macro':<12}"
print(header)
print("-"*80)

for model in models_data:
    fpr_dict, tpr_dict, roc_auc_dict = compute_roc_per_class(
        model['y_true'], model['y_prob'], N_CLASSES
    )
    _, _, auc_micro = compute_micro_roc(model['y_true'], model['y_prob'], N_CLASSES)
    _, _, auc_macro = compute_macro_roc(model['y_true'], model['y_prob'], N_CLASSES)
    
    row = f"{model['name']:<25}"
    for i in range(N_CLASSES):
        row += f"{roc_auc_dict[i]:<12.4f}"
    row += f"{auc_micro:<12.4f}{auc_macro:<12.4f}"
    print(row)

print("="*80)

#######################################################
# different graph  这几行代码均是在算法全部改进的情况下运行，只有输入的数据（点数和grph的连接方式）不同
data_19lmk_pose = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_19lmk+pose1.npz')
true_data0 = data_19lmk_pose['y_true']  # Y_test
pred_pro0 = data_19lmk_pose['y_prob']
predicted_label0 = data_19lmk_pose['y_pred']

data_20lmk_pose = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_20lmk+pose.npz')
true_data1 = data_20lmk_pose['y_true']  # Y_test
pred_pro1 = data_20lmk_pose['y_prob']
predicted_label1 = data_20lmk_pose['y_pred']

data_24lmk_pose = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_24lmk+pose.npz')
true_data2 = data_24lmk_pose['y_true']  # Y_test
pred_pro2 = data_24lmk_pose['y_prob']
predicted_label2 = data_24lmk_pose['y_pred']

data_26lmk_pose = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_26lmk+pose.npz')
true_data3 = data_26lmk_pose['y_true']  # Y_test
pred_pro3 = data_26lmk_pose['y_prob']
predicted_label3 = data_26lmk_pose['y_pred']

data_28lmk_pose = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+simi+gat+filt1.npz')
true_data4 = data_28lmk_pose['y_true']  # Y_test
pred_pro4 = data_28lmk_pose['y_prob']
predicted_label4 = data_28lmk_pose['y_pred']

data_33lmk_pose = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_33lmk+pose.npz')
true_data5 = data_33lmk_pose['y_true']  # Y_test
pred_pro5 = data_33lmk_pose['y_prob']
predicted_label5 = data_33lmk_pose['y_pred']

data_68lmk_pose = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_68lmk+pose.npz')
true_data6 = data_68lmk_pose['y_true']  # Y_test
pred_pro6 = data_68lmk_pose['y_prob']
predicted_label6 = data_68lmk_pose['y_pred']

# no pose
data_28lmk = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+nopose.npz')
true_data7 = data_28lmk['y_true']  # Y_test
pred_pro7 = data_28lmk['y_prob']
predicted_label7 = data_28lmk['y_pred']

# ========== 计算 accuracy precision recall F1 ==========
print('19lmk_pose:')
print(classification_report(true_data0, predicted_label0, digits=4))
measure_performance(true_data0, predicted_label0, predicted_label0)

print('20lmk_pose:')
print(classification_report(true_data1, predicted_label1, digits=4))
measure_performance(true_data1, predicted_label1, predicted_label1)

print('24lmk_pose:')
print(classification_report(true_data2, predicted_label2, digits=4))
measure_performance(true_data2, predicted_label2, predicted_label2)

print('26lmk_pose:')
print(classification_report(true_data3, predicted_label3, digits=4))
measure_performance(true_data3, predicted_label3, predicted_label3)

print('28lmk_pose:')
print(classification_report(true_data4, predicted_label4, digits=4))
measure_performance(true_data4, predicted_label4, predicted_label4)

print('33lmk_pose:')
print(classification_report(true_data5, predicted_label5, digits=4))
measure_performance(true_data5, predicted_label5, predicted_label5)

print('68lmk_pose:')
print(classification_report(true_data6, predicted_label6, digits=4))
measure_performance(true_data6, predicted_label6, predicted_label6)

print('28lmk no pose:')
print(classification_report(true_data7, predicted_label7, digits=4))
measure_performance(true_data7, predicted_label7, predicted_label7)

#######################################################
# abaltion study
# 95.61(在st_gcn_7的框架下去掉各个模块)， 1： 85.9（直接使用st_gcn_0运行,早停，可能陷入了局部最优） 2: 91.57（直接使用st_gcn_0运行）
data_onlykalm = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+filt2.npz')
true_data0 = data_onlykalm['y_true']  # Y_test
pred_pro0 = data_onlykalm['y_prob']
predicted_label0 = data_onlykalm['y_pred']

data_onlysimilarity = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conference/results/test_results_19lmk+pose_stgcn5_onlypose2.npz')
true_data1 = data_onlysimilarity['y_true']  # Y_test
pred_pro1= data_onlysimilarity['y_prob']
predicted_label1 = data_onlysimilarity['y_pred']

data_onlytrans = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conference/results/test_results_19lmk+pose_stgcn5_onlypose2.npz')
true_data2 = data_onlytrans['y_true']  # Y_test
pred_pro2 = data_onlytrans['y_prob']
predicted_label2 = data_onlytrans['y_pred']

data_onlygat = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conference/results/test_results_19lmk+pose_stgcn5_onlypose2.npz')
true_data3 = data_onlygat['y_true']  # Y_test
pred_pro3 = data_onlygat['y_prob']
predicted_label3 = data_onlygat['y_pred']

data_kalm_simi = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+simi+filt.npz')
                          # 
true_data4 = data_kalm_simi['y_true']  # Y_test
pred_pro4 = data_kalm_simi['y_prob']
predicted_label4 = data_kalm_simi['y_pred']

data_kalm_trans = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+filt.npz')
true_data5 = data_kalm_trans['y_true']  # Y_test
pred_pro5 = data_kalm_trans['y_prob']
predicted_label5 = data_kalm_trans['y_pred']

data_kalm_gat = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+gat+filt.npz')
true_data6 = data_kalm_gat['y_true']  # Y_test
pred_pro6 = data_kalm_gat['y_prob']
predicted_label6 = data_kalm_gat['y_pred']

data_simi_trans = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conference/results/test_results_19lmk+pose_stgcn5_onlypose2.npz')
true_data7 = data_simi_trans['y_true']  # Y_test
pred_pro7 = data_simi_trans['y_prob']
predicted_label7 = data_simi_trans['y_pred']

data_simi_gat = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conference/results/test_results_19lmk+pose_stgcn5_onlypose2.npz')
true_data8 = data_simi_gat['y_true']  # Y_test
pred_pro8 = data_simi_gat['y_prob']
predicted_label8 = data_simi_gat['y_pred']

data_trans_gat = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+gat.npz')
true_data9 = data_trans_gat['y_true']  # Y_test
pred_pro9 = data_trans_gat['y_prob']
predicted_label9 = data_trans_gat['y_pred']

data_nokalm = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+simi+trans+gat.npz')
true_data10 = data_nokalm['y_true']  # Y_test
pred_pro10 = data_nokalm['y_prob']
predicted_label10 = data_nokalm['y_pred']

data_nosimilarity = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+gat+filt.npz')
true_data11 = data_nosimilarity['y_true']  # Y_test
pred_pro11 = data_nosimilarity['y_prob']
predicted_label11 = data_nosimilarity['y_pred']

data_notransform = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+simi+gat+filt.npz')
true_data12 = data_notransform['y_true']  # Y_test
pred_pro12 = data_notransform['y_prob']
predicted_label12 = data_notransform['y_pred']

data_nogat = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+simi+filt.npz')
true_data13 = data_nogat['y_true']  # Y_test
pred_pro13 = data_nogat['y_prob']
predicted_label13 = data_nogat['y_pred']

# 95.87, 1: 97.59
data_all = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+simi+gat+filt1.npz')
true_data14 = data_all['y_true']  # Y_test
pred_pro14 = data_all['y_prob']
predicted_label14 = data_all['y_pred']

# ========== 计算 accuracy precision recall F1 ==========
print('only kalm:')
print(classification_report(true_data0, predicted_label0, digits=4))
measure_performance(true_data0, predicted_label0, predicted_label0)

print('only similarity:')
print(classification_report(true_data1, predicted_label1, digits=4))
measure_performance(true_data1, predicted_label1, predicted_label1)

print('only transform:')
print(classification_report(true_data2, predicted_label2, digits=4))
measure_performance(true_data2, predicted_label2, predicted_label2)

print('only gat:')
print(classification_report(true_data3, predicted_label3, digits=4))
measure_performance(true_data3, predicted_label3, predicted_label3)

print('kalm_simi:')
print(classification_report(true_data4, predicted_label4, digits=4))
measure_performance(true_data4, predicted_label4, predicted_label4)
    
print('kalm_trans:')
print(classification_report(true_data5, predicted_label5, digits=4))
measure_performance(true_data5, predicted_label5, predicted_label5)

print('kalm_gat:')
print(classification_report(true_data6, predicted_label6, digits=4))
measure_performance(true_data6, predicted_label6, predicted_label6)
print('simi_trans:')
print(classification_report(true_data7, predicted_label7, digits=4))
measure_performance(true_data7, predicted_label7, predicted_label7)

print('simi_gat:')
print(classification_report(true_data8, predicted_label8, digits=4))
measure_performance(true_data8, predicted_label8, predicted_label8)
print('trans_gat:')
print(classification_report(true_data9, predicted_label9, digits=4))
measure_performance(true_data9, predicted_label9, predicted_label9)

print('no kalm:')
print(classification_report(true_data10, predicted_label10, digits=4))
measure_performance(true_data10, predicted_label10, predicted_label10)

print('no similarity:')
print(classification_report(true_data11, predicted_label11, digits=4))
measure_performance(true_data11, predicted_label11, predicted_label11)

print('no transform:')
print(classification_report(true_data12, predicted_label12, digits=4))
measure_performance(true_data12, predicted_label12, predicted_label12)

print('no gat:')
print(classification_report(true_data13, predicted_label13, digits=4))
measure_performance(true_data13, predicted_label13, predicted_label13)

print('all')
print(classification_report(true_data14, predicted_label14, digits=4))
measure_performance(true_data14, predicted_label14, predicted_label14)



# 20260516的结果，新一轮的abaltion study结果

# test_results_28lmk+pose+filt2.npz 91.57
# result_journal/test_results_28lmk+pose+filt3.npz  89.8 直接使用st_gcn_0运行 temporal_kernel_size = 9  来源代码的设置一模一样
# result_journal/test_results_28lmk+pose+filt4.npz  92.03 直接使用st_gcn_0运行 temporal_kernel_size = 3  
# result_journal/test_results_28lmk+pose+filt5.npz  94.2  直接使用st_gcn_7 去掉所有的改进模块运行
# data_onlyfusion = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+filt4.npz')
data_onlyfusion = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans8.npz')
true_data0 = data_onlyfusion['y_true']  # Y_test
pred_pro0 = data_onlyfusion['y_prob']
predicted_label0 = data_onlyfusion['y_pred']

# test_results_28lmk+pose+trans1.npz 95.96
# test_results_28lmk+pose+trans2.npz 94.58
# test_results_28lmk+pose+trans3.npz 93.68 直接使用st_gcn_0运行 temporal_kernel_size = 3 + transform机制
# test_results_28lmk+pose+trans4.npz 89.25   直接使用st_gcn_0运行 temporal_kernel_size = 9 + transform机制
# test_results_28lmk+pose+trans5.npz 94.58 直接使用st_gcn_7运行 temporal_kernel_size = 9 + transform机制
# test_results_28lmk+pose+trans6.npz 96.38 直接使用st_gcn_7运行 temporal_kernel_size = 3 + transform机制
# test_results_28lmk+pose+trans7.npz 93.55 直接使用st_gcn_7运行 temporal_kernel_size = 9 + transform机制
# test_results_28lmk+pose+trans8.npz 92.26 直接使用st_gcn_0运行 temporal_kernel_size = 3 + transform机制
#  88.82 没有保存记录 直接使用st_gcn_0运行 temporal_kernel_size = 9 + transform机制
data_onlytrans = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans7.npz') 
true_data1 = data_onlytrans['y_true']  # Y_test
pred_pro1 = data_onlytrans['y_prob']
predicted_label1 = data_onlytrans['y_pred']

# test_results_28lmk+pose+simi-atten.npz 95.18
# test_results_28lmk+pose+simi-atten1.npz 95.61
# test_results_28lmk+pose+simi-atten2.npz 95.01
data_onlysimiatten = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+simi-atten2.npz') 
true_data2 = data_onlysimiatten['y_true']  # Y_test
pred_pro2= data_onlysimiatten['y_prob']
predicted_label2 = data_onlysimiatten['y_pred']

data_fusion_trans = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+filt.npz')
true_data3 = data_fusion_trans['y_true']  # Y_test
pred_pro3 = data_fusion_trans['y_prob']
predicted_label3 = data_fusion_trans['y_pred']

# test_results_28lmk+pose+fusion+simi-atten1.npz 97.25
# test_results_28lmk+pose+fusion+simi-atten2.npz 94.75
# test_results_28lmk+pose+fusion+simi-atten3.npz 94.92
# test_results_28lmk+pose+fusion+simi-atten4.npz 96.13
# 95.87  没存结果
data_fusion_simiatten = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+fusion+simi-atten4.npz')  # 93.37
true_data4 = data_fusion_simiatten['y_true']  # Y_test
pred_pro4 = data_fusion_simiatten['y_prob']
predicted_label4 = data_fusion_simiatten['y_pred']

# test_results_28lmk+pose+trans_simi-atten1.npz 94.75
# test_results_28lmk+pose+trans_simi-atten2.npz 95.18
# test_results_28lmk+pose+trans_simi-atten3.npz 95.53
# test_results_28lmk+pose+trans_simi-atten4.npz 95.79
# test_results_28lmk+pose+trans_simi-atten5.npz 96.30
data_trans_simiatten = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans_simi-atten5.npz')  # 92.86
true_data5 = data_trans_simiatten['y_true']  # Y_test
pred_pro5 = data_trans_simiatten['y_prob']
predicted_label5 = data_trans_simiatten['y_pred']

# 95.18
data_all = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+simi+gat+filt1.npz')
true_data6 = data_all['y_true']  # Y_test
pred_pro6 = data_all['y_prob']
predicted_label6 = data_all['y_pred']

print('only fusion:')
print(classification_report(true_data0, predicted_label0, digits=4))
measure_performance(true_data0, predicted_label0, predicted_label0)

# only fusion:
cm0 = confusion_matrix(true_data0, predicted_label0)
plot_confusion_picture(cm0)

print('only transform:')
print(classification_report(true_data1, predicted_label1, digits=4))
measure_performance(true_data1, predicted_label1, predicted_label1)

cm1 = confusion_matrix(true_data1, predicted_label1)
plot_confusion_picture(cm1)

print('only similarity atten:')
print(classification_report(true_data2, predicted_label2, digits=4))
measure_performance(true_data2, predicted_label2, predicted_label2)

cm2 = confusion_matrix(true_data2, predicted_label2)
plot_confusion_picture(cm2)

print('fusion_transform:')
print(classification_report(true_data3, predicted_label3, digits=4))
measure_performance(true_data3, predicted_label3, predicted_label3)

cm3 = confusion_matrix(true_data3, predicted_label3)
plot_confusion_picture(cm3)

print('fusion_similarity_atten:')
print(classification_report(true_data4, predicted_label4, digits=4))
measure_performance(true_data4, predicted_label4, predicted_label4)

cm4 = confusion_matrix(true_data4, predicted_label4)
plot_confusion_picture(cm4)
    
print('transform_similarity_atten:')
print(classification_report(true_data5, predicted_label5, digits=4))
measure_performance(true_data5, predicted_label5, predicted_label5)

cm5 = confusion_matrix(true_data5, predicted_label5)
plot_confusion_picture(cm5)

print('all:')
print(classification_report(true_data6, predicted_label6, digits=4))
measure_performance(true_data6, predicted_label6, predicted_label6)


# 运行不同的Sequence length和stride的acc，pre，recall，F1
# Sequence length:5, stride:3
data_5_3 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T5_stride3_3.npz')
true_data0 = data_5_3['y_true']  # Y_test
pred_pro0 = data_5_3['y_prob']
predicted_label0 = data_5_3['y_pred']

# Sequence length:8, stride:3
data_8_3 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T8_stride3_1.npz') 
true_data1 = data_8_3['y_true']  # Y_test
pred_pro1 = data_8_3['y_prob']
predicted_label1 = data_8_3['y_pred']

# Sequence length:15, stride:5
data_15_5 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T15_stride5_1.npz') 
true_data2 = data_15_5['y_true']  # Y_test
pred_pro2= data_15_5['y_prob']
predicted_label2 = data_15_5['y_pred']

# Sequence length:23, stride:8
data_23_8 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T23_stride8.npz')
true_data3 = data_23_8['y_true']  # Y_test
pred_pro3 = data_23_8['y_prob']
predicted_label3 = data_23_8['y_pred']

# Sequence length:30, stride:10
data_30_10 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+simi+gat+filt1.npz')  
true_data4 = data_30_10['y_true']  # Y_test
pred_pro4 = data_30_10['y_prob']
predicted_label4 = data_30_10['y_pred']

# Sequence length:34, stride:12
data_34_12 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T34_stride12.npz')  
true_data5 = data_34_12['y_true']  # Y_test
pred_pro5 = data_34_12['y_prob']
predicted_label5 = data_34_12['y_pred']

# Sequence length:38, stride:13
data_38_13 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T38_stride13_1.npz')
true_data6 = data_38_13['y_true']  # Y_test
pred_pro6 = data_38_13['y_prob']
predicted_label6 = data_38_13['y_pred']

print('Sequence length:5, stride:3')
print(classification_report(true_data0, predicted_label0, digits=4))
measure_performance(true_data0, predicted_label0, predicted_label0)

print('Sequence length:8, stride:3')
print(classification_report(true_data1, predicted_label1, digits=4))
measure_performance(true_data1, predicted_label1, predicted_label1)

print('Sequence length:15, stride:5')
print(classification_report(true_data2, predicted_label2, digits=4))
measure_performance(true_data2, predicted_label2, predicted_label2)

print('Sequence length:23, stride:8')
print(classification_report(true_data3, predicted_label3, digits=4))
measure_performance(true_data3, predicted_label3, predicted_label3)

print('Sequence length:30, stride:10')
print(classification_report(true_data4, predicted_label4, digits=4))
measure_performance(true_data4, predicted_label4, predicted_label4)
    
print('Sequence length:34, stride:12')
print(classification_report(true_data5, predicted_label5, digits=4))
measure_performance(true_data5, predicted_label5, predicted_label5)

print('Sequence length:38, stride:13')
print(classification_report(true_data6, predicted_label6, digits=4))
measure_performance(true_data6, predicted_label6, predicted_label6)


"""
plot_adjacency_heatmaps.py
用法: python plot_adjacency_heatmaps.py --data_dir work_dir/adjacency_data
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


# =========================
# 全局字体设置
# =========================
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


def plot_single_heatmap(matrix, title, save_path,
                        cmap='viridis', vmin=None, vmax=None,
                        figsize=(8, 7), annot=False, fmt='.2f',
                        node_labels=None, center=None):
    """绘制单张 heatmap 并保存"""
    fig, ax = plt.subplots(figsize=figsize)

    kwargs = dict(
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        square=True,
        xticklabels=node_labels if node_labels is not None else True,
        yticklabels=node_labels if node_labels is not None else True,
        ax=ax,
        cbar_kws={'shrink': 0.8},
    )

    if vmin is not None:
        kwargs['vmin'] = vmin
    if vmax is not None:
        kwargs['vmax'] = vmax
    if center is not None:
        kwargs['center'] = center

    hm = sns.heatmap(matrix, **kwargs)

    # 标题和坐标轴
    ax.set_title(title, fontsize=24, fontname='Times New Roman')
    ax.set_xlabel('Landmarks', fontsize=20, fontname='Times New Roman')
    ax.set_ylabel('Landmarks', fontsize=20, fontname='Times New Roman')

    # 统一刻度方向
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center',
             fontsize=12, fontname='Times New Roman')
    plt.setp(ax.get_yticklabels(), rotation=0,
             fontsize=12, fontname='Times New Roman')

    # colorbar 字体
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    plt.tight_layout()
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")



def plot_diff_topk(diff, save_path, title='A_eff Difference (Top-k Highlighted)',
                   topk=12, figsize=(8, 7), node_labels=None):
    """绘制差值图并高亮绝对值最大的 Top-k 元素"""
    fig, ax = plt.subplots(figsize=figsize)

    hm = sns.heatmap(
        diff,
        cmap='RdBu_r',
        center=0,
        square=True,
        xticklabels=node_labels if node_labels is not None else True,
        yticklabels=node_labels if node_labels is not None else True,
        ax=ax,
        cbar_kws={'shrink': 0.8},
    )

    ax.set_title(title, fontsize=24, fontname='Times New Roman')
    ax.set_xlabel('Landmarks', fontsize=20, fontname='Times New Roman')
    ax.set_ylabel('Landmarks', fontsize=20, fontname='Times New Roman')

    # 统一刻度方向
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center',
             fontsize=12, fontname='Times New Roman')
    plt.setp(ax.get_yticklabels(), rotation=0,
             fontsize=12, fontname='Times New Roman')

    # colorbar 字体
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # Top-k 绝对值最大位置：计算方式保持不变
    flat_idx = np.argsort(np.abs(diff).flatten())[-topk:]
    rows, cols = np.unravel_index(flat_idx, diff.shape)

    # 不需要高亮的位置
    skip_positions = {(17, 22)}   # 如果也想跳过对称位置，就写成 {(17,22), (22,17)}

    # 画框 + 标数值
    for r, c in zip(rows, cols):
        if (r, c) in skip_positions:
            continue

        rect = patches.Rectangle(
            (c, r), 1, 1,
            linewidth=1.8,
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(rect)

        ax.text(
            c + 0.5, r + 0.5,
            f'{diff[r, c]:.2f}',
            ha='center',
            va='center',
            fontsize=7,
            color='black',
            fontname='Times New Roman'
        )

    plt.tight_layout()
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/Adjacency_Matrix/connection2/new/',
        # default='/home/zhangxuan/sourcecode/EVA-GCN-conf-based/work_dir_19_points_FAN_6_layers_train_on_300W_LP_test_on_BIWI/GCN_head_pose/20260116/adjacency_data',
        help='存放 .npy 文件的目录'
    )
    parser.add_argument(
        '--cmap_A',
        type=str,
        default='Blues',
        help='物理连接图颜色'
    )
    parser.add_argument(
        '--cmap_eff',
        type=str,
        default='Reds',
        help='A_eff 颜色'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='图片输出目录，默认与 data_dir 相同'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=33,
        help='差值图中高亮绝对值最大的前 k 个位置'
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # 1. 加载原始物理连接图
    # =========================
    A_path = os.path.join(args.data_dir, 'A_physical.npy')
    if not os.path.exists(A_path):
        raise FileNotFoundError(f'File not found: {A_path}')

    A = np.load(A_path)
    print(f"Loaded A_physical: shape={A.shape}")

    # 如果是 (K, V, V)，合并为 (V, V)
    if A.ndim == 3:
        A_plot = A.sum(axis=0)
    elif A.ndim == 2:
        A_plot = A
    else:
        raise ValueError(f'Unexpected shape for A_physical: {A.shape}')

    num_nodes = A_plot.shape[0]
    node_labels = [str(i) for i in range(num_nodes)]

    plot_single_heatmap(
        A_plot,
        title='Adjacency Matrix (Initial)',
        save_path=os.path.join(output_dir, 'heatmap_A_physical.png'),
        cmap=args.cmap_A,
        vmin=0,
        vmax=0.5,
        node_labels=node_labels
    )

    # =========================
    # 2. 训练前 A_eff
    # =========================
    before_path = os.path.join(args.data_dir, 'A_eff_after_train_1.npy')
    A_eff_before_plot = None

    if os.path.exists(before_path):
        A_eff_before = np.load(before_path)
        print(f"Loaded A_eff_before_train: shape={A_eff_before.shape}")

        if A_eff_before.ndim == 3:
            A_eff_before_plot = A_eff_before.sum(axis=0)
        elif A_eff_before.ndim == 2:
            A_eff_before_plot = A_eff_before
        else:
            raise ValueError(f'Unexpected shape for A_eff_before: {A_eff_before.shape}')

        plot_single_heatmap(
            A_eff_before_plot,
            title='A_eff Before Training',
            save_path=os.path.join(output_dir, 'heatmap_A_eff_after_train_1.png'),
            cmap=args.cmap_eff,
            node_labels=node_labels
        )
    else:
        print(f'Not found: {before_path}')

    # =========================
    # 3. 训练后 A_eff
    # =========================
    after_path = os.path.join(args.data_dir, 'A_eff_after_train_200.npy')
    A_eff_after_plot = None

    if os.path.exists(after_path):
        A_eff_after = np.load(after_path)
        print(f"Loaded A_eff_after_train: shape={A_eff_after.shape}")

        if A_eff_after.ndim == 3:
            A_eff_after_plot = A_eff_after.sum(axis=0)
        elif A_eff_after.ndim == 2:
            A_eff_after_plot = A_eff_after
        else:
            raise ValueError(f'Unexpected shape for A_eff_after: {A_eff_after.shape}')

        plot_single_heatmap(
            A_eff_after_plot,
            title='Adjacency Matrix (After Training)',
            save_path=os.path.join(output_dir, 'heatmap_A_eff_after_train_200.png'),
            cmap=args.cmap_eff,
            node_labels=node_labels
        )
    else:
        print(f'Not found: {after_path}')

    # =========================
    # 4. 差值图
    #    优先：训练后 - 原始物理图
    # =========================
    if A_eff_after_plot is not None:
        diff = A_eff_after_plot - A_plot

        plot_single_heatmap(
            diff,
            title='A_eff Difference',
            save_path=os.path.join(output_dir, 'heatmap_A_eff_diff.png'),
            cmap='RdBu_r',
            center=0,
            node_labels=node_labels
        )

        plot_diff_topk(
            diff,
            save_path=os.path.join(output_dir, 'heatmap_A_eff_diff_topk.png'),
            title=f'Adjacency Matrix Difference (Top-32 Highlighted)', # {args.topk}
            topk=args.topk,
            node_labels=node_labels
        )

    # =========================
    # 5. 可选：训练后 - 训练前
    # =========================
    if (A_eff_before_plot is not None) and (A_eff_after_plot is not None):
        diff_train = A_eff_after_plot - A_eff_before_plot

        plot_single_heatmap(
            diff_train,
            title='A_eff Difference (After - Before)',
            save_path=os.path.join(output_dir, 'heatmap_A_eff_diff_after_minus_before.png'),
            cmap='RdBu_r',
            center=0,
            node_labels=node_labels
        )

        plot_diff_topk(
            diff_train,
            save_path=os.path.join(output_dir, 'heatmap_A_eff_diff_after_minus_before_topk.png'),
            title=f'A_eff Difference After-Before (Top-{args.topk} Highlighted)',
            topk=args.topk,
            node_labels=node_labels
        )


if __name__ == '__main__':
    main()


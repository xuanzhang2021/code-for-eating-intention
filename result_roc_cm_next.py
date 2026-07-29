import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# 这个文件是在roc_cm的基础上，继续补充的实验，里面使用‘’‘’‘’注释的内容不能删除
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
'''
############ 以下是sequence length
# 95.74
# 1 95.32
# 2 94.80
# 3 94.45
data0 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T5_stride3_2.npz')   
true_data0 = data0['y_true']
pred_pro0 = data0['y_prob']
predicted_label0 = data0['y_pred']

# 96.26
# 1 95.71
data1 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T8_stride3_1.npz')
true_data1 = data1['y_true']
pred_pro1 = data1['y_prob']
predicted_label1 = data1['y_pred']

# 94.41
# 1 95.77
data2 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T15_stride5_1.npz')
true_data2 = data2['y_true']
pred_pro2 = data2['y_prob']
predicted_label2 = data2['y_pred']

# 96.12
data3 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T23_stride8.npz')
true_data3 = data3['y_true']
pred_pro3 = data3['y_prob']
predicted_label3 = data3['y_pred']

# 95.01
data4 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T34_stride12.npz')
true_data4 = data4['y_true']
pred_pro4 = data4['y_prob']
predicted_label4 = data4['y_pred']

# 93.55
data5 = np.load('/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T38_stride13_1.npz')
true_data5 = data5['y_true']
pred_pro5 = data5['y_prob']
predicted_label5 = data5['y_pred']


print('5 3')
print(classification_report(true_data0, predicted_label0, digits=4))
measure_performance(true_data0, predicted_label0, predicted_label0)

print('8 3')
print(classification_report(true_data1, predicted_label1, digits=4))
measure_performance(true_data1, predicted_label1, predicted_label1)

print('15 5')
print(classification_report(true_data2, predicted_label2, digits=4))
measure_performance(true_data2, predicted_label2, predicted_label2)

print('23 8')
print(classification_report(true_data3, predicted_label3, digits=4))
measure_performance(true_data3, predicted_label3, predicted_label3)

print('34 12')
print(classification_report(true_data4, predicted_label4, digits=4))
measure_performance(true_data4, predicted_label4, predicted_label4)

print('38 13')
print(classification_report(true_data5, predicted_label5, digits=4))
measure_performance(true_data5, predicted_label5, predicted_label5)
#################### 以上是sequence length

#########################以下是求sample level 的mean and std
'''

def evaluate_multiple_runs(npz_files, average='macro'):  # average= 'weighted' or macro, 整篇论文中我用的macro
    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []

    for file in npz_files:
        data = np.load(file)
        y_true = data['y_true']
        y_pred = data['y_pred']

        acc = accuracy_score(y_true, y_pred)
        pre, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )

        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)

        print(f'{file}')
        print(f'  Acc:    {acc*100:.2f}%')
        print(f'  Pre:    {pre*100:.2f}%')
        print(f'  Recall: {rec*100:.2f}%')
        print(f'  F1:     {f1*100:.2f}%')
        print('-' * 50)

    acc_arr = np.array(acc_list) * 100
    pre_arr = np.array(pre_list) * 100
    rec_arr = np.array(rec_list) * 100
    f1_arr = np.array(f1_list) * 100

    print('\nFinal Results ({})'.format(average))
    print(f'Acc(%):    {acc_arr.mean():.2f} ± {acc_arr.std(ddof=1):.2f}')
    print(f'Pre(%):    {pre_arr.mean():.2f} ± {pre_arr.std(ddof=1):.2f}')
    print(f'Recall(%): {rec_arr.mean():.2f} ± {rec_arr.std(ddof=1):.2f}')
    print(f'F1(%):     {f1_arr.mean():.2f} ± {f1_arr.std(ddof=1):.2f}')

    return acc_arr, pre_arr, rec_arr, f1_arr


print("resnet-lstm")
npz_files = [
    '/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/tie_result/sequence_level_predictions_20260421_122615_resnet_lstm.npz',  # 91.23
    '/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/tie_result/sequence_level_predictions_20260519_220534_resnet_lstm.npz',   # 90.89
    '/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/case_result/sequence_level_predictions_20260209_215208.npz',
]
evaluate_multiple_runs(npz_files)

print("resnet-lstm with guassian")
npz_files = [
    '/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/tie_result/sequence_level_predictions_20260421_120551_gaussian.npz',  # 91.23
    '/home/zhangxuan/sourcecode/LSTM-Neural-Network-for-Time-Series-Prediction-master/Stacked_Bi_Uni_LSTM/Keras_LSTM/case_result/sequence_level_predictions_20260209_190237.npz',
    
]
evaluate_multiple_runs(npz_files)

print("stgcn")
npz_files = [
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn1.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn2.npz',
]
evaluate_multiple_runs(npz_files)

print("2sgcn")
npz_files = [
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_2sagcn.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_2sagcn1.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_2sagcn2.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_2sagcn3.npz',
]
evaluate_multiple_runs(npz_files)

print("msg3d")
npz_files = [
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_msg3d.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_msg3d1.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_msg3d2.npz',
]
evaluate_multiple_runs(npz_files)

print("ctrgcn")
npz_files = [
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_ctrgcn.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_ctrgcn1.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_ctrgcn2.npz',
    # '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_ctrgcn3.npz',
]
evaluate_multiple_runs(npz_files)

print("sttformer")
npz_files = [
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_sttformer.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_sttformer1.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_sttformer2.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_sttformer3.npz',
]
evaluate_multiple_runs(npz_files)

print("skateformer")
npz_files = [
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_skateformer.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_skateformer1.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_skateformer2.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn7_skateformer3.npz',
]
evaluate_multiple_runs(npz_files)

print("stgcn7")
npz_files = [
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/window_size/test_results_28lmk+pose_stgcn7_T30_stride10.npz',
    # '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+simi+gat+filt.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose+trans+simi+gat+filt1.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn8.npz',
    '/home/zhangxuan/sourcecode/EVA-GCN-conf-based/result_journal/test_results_28lmk+pose_stgcn81.npz',
]
evaluate_multiple_runs(npz_files)

########################以上是求sample level 的mean and std
resnet = np.array([91.23, 90.89, 90.43, 90.89])
resnet_gaussian = np.array([94.75, 93.70, 94.75, 93.70])
stgcn = np.array([90.71, 88.91, 91.23, 90.71])
sAGCN = np.array([95.36, 90.63, 91.83, 92.52])
msg3d = np.array([94.71, 98.02, 96.76, 95.71])
ctrgcn = np.array([95.96, 96.13, 96.82, 96.13])
STTFormer = np.array([93.81, 93.55, 91.75, 91.92])
SkateFormer = np.array([95.10, 94.50, 93.72, 94.24])
stgcn7 = np.array([96.54, 97.59, 98.97, 97.59], dtype=float)

baselines = {
    "ResNet": resnet,
    "resnet_gaussian": resnet_gaussian,
    "STGCN": stgcn,
    "2s-AGCN": sAGCN,
    "MSG3D" : msg3d,
    "CTR-GCN": ctrgcn,
    "STTFormer": STTFormer,
    "SkateFormer": SkateFormer
}

results = []

for name, scores in baselines.items():
    t_stat, t_p = ttest_rel(stgcn7, scores)
    w_stat, w_p = wilcoxon(stgcn7, scores)
    
    results.append({
        "Comparison": f"STGCN7 vs {name}",
        "t_stat": t_stat,
        "t_p": t_p,
        "w_stat": w_stat,
        "w_p": w_p
    })

df = pd.DataFrame(results)
print(df)

MSG3D_acc = np.array([95.71, 98.02, 96.76, 95.71])
MSG3D_pre = np.array([95.96, 98.19, 96.92, 95.96])
MSG3D_rec = np.array([95.74, 97.76, 96.57, 95.74])
MSG3D_f1 = np.array([95.84, 97.96, 96.73, 95.85])

print(f'Acc(%):    {MSG3D_acc.mean():.2f} ± {MSG3D_acc.std(ddof=1):.2f}')
print(f'Pre(%):    {MSG3D_pre.mean():.2f} ± {MSG3D_pre.std(ddof=1):.2f}')
print(f'Recall(%): {MSG3D_rec.mean():.2f} ± {MSG3D_rec.std(ddof=1):.2f}')
print(f'F1(%):     {MSG3D_f1.mean():.2f} ± {MSG3D_f1.std(ddof=1):.2f}')


# stgcn7_acc = np.array([96.04, 97.59, 98.97, 98.97], dtype=float)
# stgcn7_pre = np.array([96.22, 97.81, 99.03, 99.07], dtype=float)
# stgcn7_rec = np.array([95.83, 97.53, 98.73, 98.78], dtype=float)
# stgcn7_f1 = np.array([96.01, 97.66,98.87, 98.92], dtype=float)

stgcn7_acc = np.array([96.54, 97.59, 98.97, 97.59], dtype=float)
stgcn7_pre = np.array([96.72, 97.81, 99.03, 97.81], dtype=float)
stgcn7_rec = np.array([96.33, 97.53, 98.73, 97.53], dtype=float)
stgcn7_f1 = np.array([96.51, 97.66, 98.87, 97.66], dtype=float)


print(f'Acc(%):    {stgcn7_acc.mean():.2f} ± {stgcn7_acc.std(ddof=1):.2f}')
print(f'Pre(%):    {stgcn7_pre.mean():.2f} ± {stgcn7_pre.std(ddof=1):.2f}')
print(f'Recall(%): {stgcn7_rec.mean():.2f} ± {stgcn7_rec.std(ddof=1):.2f}')
print(f'F1(%):     {stgcn7_f1.mean():.2f} ± {stgcn7_f1.std(ddof=1):.2f}')





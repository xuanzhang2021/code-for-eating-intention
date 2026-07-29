import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Patch

# =========================
# 0. matplotlib论文风格
# =========================
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 300

# 字体：论文常用 serif + Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 全局字号：适合论文，不要太大
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# 线宽与样式
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5

# seaborn风格
sns.set_style("white")

# =========================
# 0.1 统一样式参数
# =========================
FONT_NAME = 'Times New Roman'
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 9
LEGEND_SIZE = 9
ANNOT_SIZE = 8

# =========================
# 0.2 heatmap高亮配置
# =========================
HEATMAP_HIGHLIGHT_TOPK = 21
HEATMAP_HIGHLIGHT_COLOR = 'white'   # 论文风格更克制
HEATMAP_HIGHLIGHT_LINEWIDTH = 1.0
HEATMAP_EXCLUDE_DIAGONAL = False

# =========================
# 1. 路径设置
# =========================
work_dir = './work_dir_19_points_FAN_6_layers_train_on_300W_LP_test_on_BIWI/GCN_head_pose/20260116/attn'

alpha_path = os.path.join(work_dir, 'test_alpha_x.npy')
label_path = os.path.join(work_dir, 'test_labels.npy')
idx_path = os.path.join(work_dir, 'test_sample_idx.npy')
result_path = os.path.join(work_dir, 'test_results.npz')

save_dir = os.path.join(work_dir, 'attn_vis')
os.makedirs(save_dir, exist_ok=True)

# =========================
# 2. 读取数据
# =========================
alpha_x = np.load(alpha_path)      # [N, T, V, V]
labels = np.load(label_path)       # [N]

sample_idx = None
if os.path.exists(idx_path):
    sample_idx = np.load(idx_path)

results = None
if os.path.exists(result_path):
    results = np.load(result_path)
    y_true = results['y_true']
    y_pred = results['y_pred']
    y_prob = results['y_prob'] if 'y_prob' in results.files else None
else:
    y_true = labels
    y_pred = None
    y_prob = None

print('alpha_x shape:', alpha_x.shape)
print('labels shape :', labels.shape)

if sample_idx is not None:
    print('sample_idx shape:', sample_idx.shape)

# =========================
# 3. 安全检查
# =========================
assert alpha_x.ndim == 4, f'Expect alpha_x shape [N,T,V,V], but got {alpha_x.shape}'
assert labels.ndim == 1, f'Expect labels shape [N], but got {labels.shape}'
assert alpha_x.shape[0] == labels.shape[0], 'Sample number mismatch between alpha_x and labels'

N, T, V, V2 = alpha_x.shape
assert V == V2, f'Last two dims must be equal, but got {V}, {V2}'

if y_true is not None:
    assert len(y_true) == N, f'y_true length mismatch: {len(y_true)} vs {N}'
if y_pred is not None:
    assert len(y_pred) == N, f'y_pred length mismatch: {len(y_pred)} vs {N}'

# =========================
# 4. 节点名称
# =========================
joint_names = [str(i) for i in range(V)]
assert len(joint_names) == V, f'joint_names length {len(joint_names)} must equal V={V}'

# =========================
# 5. 基础工具函数
# =========================
def style_axis_text(ax, title=None, xlabel=None, ylabel=None):
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE, fontname=FONT_NAME, pad=8)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontname=FONT_NAME, labelpad=6)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontname=FONT_NAME, labelpad=6)

    for label in ax.get_xticklabels():
        label.set_fontname(FONT_NAME)
        label.set_fontsize(TICK_SIZE)

    for label in ax.get_yticklabels():
        label.set_fontname(FONT_NAME)
        label.set_fontsize(TICK_SIZE)


def get_topk_coords_from_matrix(mat, topk=21, exclude_diagonal=False):
    mat = np.asarray(mat)
    assert mat.ndim == 2, f'Expect 2D matrix, got shape={mat.shape}'

    mat_for_rank = np.array(mat, copy=True)

    if exclude_diagonal:
        diag_len = min(mat_for_rank.shape[0], mat_for_rank.shape[1])
        diag_idx = np.arange(diag_len)
        mat_for_rank[diag_idx, diag_idx] = -np.inf

    flat = mat_for_rank.reshape(-1)
    valid_mask = np.isfinite(flat)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return []

    k = min(topk, len(valid_indices))
    valid_values = flat[valid_indices]

    top_local_idx = np.argsort(valid_values)[-k:][::-1]
    top_global_idx = valid_indices[top_local_idx]

    rows, cols = np.unravel_index(top_global_idx, mat_for_rank.shape)
    coords = list(zip(rows.tolist(), cols.tolist()))
    return coords


def draw_topk_rectangles(ax, mat, topk=21, color='white', linewidth=1.0,
                         exclude_diagonal=False, verbose=False, title=''):
    coords = get_topk_coords_from_matrix(mat, topk=topk, exclude_diagonal=exclude_diagonal)

    if verbose and len(coords) > 0:
        print(f'\nTop-{len(coords)} attention entries for: {title}')
        for rank, (r, c) in enumerate(coords, start=1):
            print(f'  #{rank}: source={r}, target={c}, value={mat[r, c]:.6f}')

    for r, c in coords:
        rect = Rectangle(
            (c, r), 1, 1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth
        )
        ax.add_patch(rect)


def save_heatmap(mat, title, save_path, cmap='viridis', center=None,
                 xlabel='Target landmarks', ylabel='Source landmarks',
                 xticklabels=None, yticklabels=None, annot=False,
                 vmin=None, vmax=None,
                 highlight_topk=None, highlight_color='white',
                 highlight_linewidth=1.0, exclude_diagonal=False,
                 verbose_topk=False):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        center=center,
        xticklabels=xticklabels if xticklabels is not None else True,
        yticklabels=yticklabels if yticklabels is not None else True,
        annot=annot,
        fmt='.3f' if annot else '',
        vmin=vmin,
        vmax=vmax,
        cbar=True
    )

    if highlight_topk is not None and highlight_topk > 0:
        draw_topk_rectangles(
            ax=ax,
            mat=mat,
            topk=highlight_topk,
            color=highlight_color,
            linewidth=highlight_linewidth,
            exclude_diagonal=exclude_diagonal,
            verbose=verbose_topk,
            title=title
        )

    style_axis_text(ax, title=title, xlabel=xlabel, ylabel=ylabel)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname(FONT_NAME)
        label.set_fontsize(TICK_SIZE)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_bar(values, title, save_path, xlabel='Landmark', ylabel='Importance',
             ylim=None, topk=10, annotate_topk=True, highlight_topk=True,
             xtick_labels=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(values))
    values = np.asarray(values)

    topk = min(topk, len(values))
    top_idx = np.argsort(values)[-topk:][::-1]
    top_set = set(top_idx.tolist())

    if highlight_topk:
        colors = ['tomato' if i in top_set else 'steelblue' for i in range(len(values))]
    else:
        colors = ['steelblue'] * len(values)

    ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)

    if ylim is not None:
        ax.set_ylim(ylim)
        upper = ylim[1]
    else:
        vmax = values.max()
        upper = vmax * 1.18 if vmax > 0 else 1.0
        ax.set_ylim(0, upper)

    if annotate_topk:
        text_offset = upper * 0.01 if upper > 0 else 0.001
        for rank, i in enumerate(top_idx, start=1):
            v = values[i]
            ax.text(
                i, v + text_offset,
                f'#{rank}\n{v:.4f}',
                ha='center', va='bottom',
                fontsize=ANNOT_SIZE, rotation=90, color='black',
                fontname=FONT_NAME
            )

    if xtick_labels is None:
        xtick_labels = [str(i) for i in x]

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=0, fontname=FONT_NAME, fontsize=TICK_SIZE)

    style_axis_text(ax, title=title, xlabel=xlabel, ylabel=ylabel)

    if highlight_topk:
        legend_handles = [
            Patch(facecolor='tomato', edgecolor='black', label=f'Top {topk} landmarks'),
            Patch(facecolor='steelblue', edgecolor='black', label='Other landmarks')
        ]
        leg = ax.legend(handles=legend_handles, loc='upper left', frameon=True, fontsize=LEGEND_SIZE)
        for text in leg.get_texts():
            text.set_fontname(FONT_NAME)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return top_idx


def print_topk(values, title, topk=10):
    values = np.asarray(values)
    topk = min(topk, len(values))
    top_idx = np.argsort(values)[-topk:][::-1]

    print(f'\n{title} top-{topk}:')
    for rank, idx in enumerate(top_idx, start=1):
        print(f'  #{rank}: node {idx} ({joint_names[idx]}), importance={values[idx]:.6f}')

    return top_idx


def compute_node_importance_from_alpha(alpha):
    if alpha.ndim == 3:
        return alpha.mean(axis=0).mean(axis=0)
    elif alpha.ndim == 4:
        return alpha.mean(axis=0).mean(axis=0).mean(axis=0)
    else:
        raise ValueError(f'Unsupported alpha ndim={alpha.ndim}')


def compute_mean_attention_from_alpha(alpha):
    if alpha.ndim == 3:
        return alpha.mean(axis=0)
    elif alpha.ndim == 4:
        return alpha.mean(axis=0).mean(axis=0)
    else:
        raise ValueError(f'Unsupported alpha ndim={alpha.ndim}')


def get_class_ids():
    return np.unique(labels)


def get_class_node_importance_dict():
    class_ids = get_class_ids()
    class_imp = {}
    for cid in class_ids:
        cls_idx = np.where(labels == cid)[0]
        if len(cls_idx) == 0:
            continue
        class_imp[cid] = compute_node_importance_from_alpha(alpha_x[cls_idx])
    return class_imp


def get_unified_node_importance_ylim(top_margin=1.15):
    candidates = []

    candidates.append(compute_node_importance_from_alpha(alpha_x).max())
    candidates.append(compute_node_importance_from_alpha(alpha_x[0]).max())

    class_imp_dict = get_class_node_importance_dict()
    for v in class_imp_dict.values():
        candidates.append(v.max())

    if y_pred is not None:
        correct_mask = (y_true == y_pred)
        wrong_mask = (y_true != y_pred)
        if correct_mask.sum() > 0:
            candidates.append(compute_node_importance_from_alpha(alpha_x[correct_mask]).max())
        if wrong_mask.sum() > 0:
            candidates.append(compute_node_importance_from_alpha(alpha_x[wrong_mask]).max())

    ymax = max(candidates)
    return (0, ymax * top_margin)


def get_representative_frame_id(alpha_sample, method='offdiag_sum'):
    assert alpha_sample.ndim == 3, f'Expect [T,V,V], got {alpha_sample.shape}'
    T_, V_, V2_ = alpha_sample.shape
    assert V_ == V2_

    scores = []
    for t in range(T_):
        mat = alpha_sample[t]

        if method == 'sum':
            score = mat.sum()
        elif method == 'std':
            score = mat.std()
        elif method == 'offdiag_sum':
            score = mat.sum() - np.trace(mat)
        else:
            raise ValueError(f'Unsupported method: {method}')

        scores.append(score)

    scores = np.asarray(scores)
    best_frame = int(np.argmax(scores))
    return best_frame


def select_samples_for_class(class_id, num_samples=2, prefer_correct=True):
    cls_idx = np.where(labels == class_id)[0].tolist()

    if len(cls_idx) == 0:
        print(f'Class {class_id} has no samples.')
        return []

    selected = []

    if prefer_correct and (y_pred is not None):
        correct_cls_idx = [i for i in cls_idx if y_true[i] == y_pred[i]]
        selected.extend(correct_cls_idx[:num_samples])

    if len(selected) < num_samples:
        for i in cls_idx:
            if i not in selected:
                selected.append(i)
            if len(selected) >= num_samples:
                break

    print(f'Class {class_id} selected samples: {selected}')
    return selected[:num_samples]


def get_sample_meta(sample_id):
    true_label = int(y_true[sample_id]) if y_true is not None else int(labels[sample_id])
    pred_label = int(y_pred[sample_id]) if y_pred is not None else None
    is_correct = (pred_label == true_label) if pred_label is not None else None
    return true_label, pred_label, is_correct


# =========================
# 6. 单样本可视化
# =========================
def plot_sample_frame(sample_id=0, frame_id=0):
    mat = alpha_x[sample_id, frame_id]
    true_label, pred_label, is_correct = get_sample_meta(sample_id)

    if pred_label is None:
        title = f'Sample {sample_id} | True {true_label} | Frame {frame_id} Attention'
    else:
        title = f'Sample {sample_id} | True {true_label} | Pred {pred_label} | Correct={is_correct} | Frame {frame_id} Attention'

    save_path = os.path.join(save_dir, f'sample_{sample_id}_frame_{frame_id}_attn.pdf')
    save_heatmap(
        mat, title, save_path, cmap='viridis',
        xticklabels=joint_names, yticklabels=joint_names,
        highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
        highlight_color=HEATMAP_HIGHLIGHT_COLOR,
        highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
        exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
    )
    print(f'Saved: {save_path}')


def plot_sample_mean_time(sample_id=0):
    mat = compute_mean_attention_from_alpha(alpha_x[sample_id])
    true_label, pred_label, is_correct = get_sample_meta(sample_id)

    if pred_label is None:
        title = f'Sample {sample_id} | True {true_label} | Mean Attention Over Time'
    else:
        title = f'Sample {sample_id} | True {true_label} | Pred {pred_label} | Correct={is_correct} | Mean Attention Over Time'

    save_path = os.path.join(save_dir, f'sample_{sample_id}_mean_time_attn.pdf')
    save_heatmap(
        mat, title, save_path, cmap='magma',
        xticklabels=joint_names, yticklabels=joint_names,
        highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
        highlight_color=HEATMAP_HIGHLIGHT_COLOR,
        highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
        exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
    )
    print(f'Saved: {save_path}')


def plot_sample_node_importance(sample_id=0, global_ylim=None, topk=10):
    node_imp = compute_node_importance_from_alpha(alpha_x[sample_id])
    true_label, pred_label, is_correct = get_sample_meta(sample_id)

    if pred_label is None:
        title = f'Sample {sample_id} | True {true_label} | Node Importance'
    else:
        title = f'Sample {sample_id} | True {true_label} | Pred {pred_label} | Correct={is_correct} | Node Importance'

    save_path = os.path.join(save_dir, f'sample_{sample_id}_node_importance.pdf')

    save_bar(
        node_imp, title, save_path,
        ylim=global_ylim, topk=topk,
        annotate_topk=True, highlight_topk=True,
        xtick_labels=joint_names
    )
    print(f'Saved: {save_path}')
    print_topk(node_imp, f'Sample {sample_id}', topk=topk)


def plot_one_sample_all(sample_id, frame_id=None, global_ylim=None, topk=10, class_name=None):
    if frame_id is None:
        frame_id = T // 2

    true_label, pred_label, is_correct = get_sample_meta(sample_id)

    prefix = f'sample_{sample_id}'
    if class_name is not None:
        prefix = f'class_{class_name}_sample_{sample_id}'

    mat_frame = alpha_x[sample_id, frame_id]
    if pred_label is None:
        title_frame = f'Sample {sample_id} | True {true_label} | Frame {frame_id} Attention'
    else:
        title_frame = f'Sample {sample_id} | True {true_label} | Pred {pred_label} | Correct={is_correct} | Frame {frame_id} Attention'

    save_path_frame = os.path.join(save_dir, f'{prefix}_frame_{frame_id}_attn.pdf')
    save_heatmap(
        mat_frame, title_frame, save_path_frame, cmap='viridis',
        xticklabels=joint_names, yticklabels=joint_names,
        highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
        highlight_color=HEATMAP_HIGHLIGHT_COLOR,
        highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
        exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
    )
    print(f'Saved: {save_path_frame}')

    mat_mean = compute_mean_attention_from_alpha(alpha_x[sample_id])
    if pred_label is None:
        title_mean = f'Sample {sample_id} | True {true_label} | Mean Attention Over Time'
    else:
        title_mean = f'Sample {sample_id} | True {true_label} | Pred {pred_label} | Correct={is_correct} | Mean Attention Over Time'

    save_path_mean = os.path.join(save_dir, f'{prefix}_mean_time_attn.pdf')
    save_heatmap(
        mat_mean, title_mean, save_path_mean, cmap='magma',
        xticklabels=joint_names, yticklabels=joint_names,
        highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
        highlight_color=HEATMAP_HIGHLIGHT_COLOR,
        highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
        exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
    )
    print(f'Saved: {save_path_mean}')

    node_imp = compute_node_importance_from_alpha(alpha_x[sample_id])
    if pred_label is None:
        title_bar = f'Sample {sample_id} | True {true_label} | Node Importance'
    else:
        title_bar = f'Sample {sample_id} | True {true_label} | Pred {pred_label} | Correct={is_correct} | Node Importance'

    save_path_bar = os.path.join(save_dir, f'{prefix}_node_importance.pdf')
    save_bar(
        node_imp, title_bar, save_path_bar,
        ylim=global_ylim, topk=topk,
        annotate_topk=True, highlight_topk=True,
        xtick_labels=joint_names
    )
    print(f'Saved: {save_path_bar}')
    print_topk(node_imp, f'Sample {sample_id}', topk=topk)


def plot_selected_samples_individually(
    class_ids=(0, 1),
    num_samples_per_class=2,
    topk=10,
    global_ylim=None,
    prefer_correct=True,
    frame_select_method='offdiag_sum'
):
    for cid in class_ids:
        selected_samples = select_samples_for_class(
            class_id=cid,
            num_samples=num_samples_per_class,
            prefer_correct=prefer_correct
        )

        for sid in selected_samples:
            rep_frame = get_representative_frame_id(alpha_x[sid], method=frame_select_method)
            true_label, pred_label, is_correct = get_sample_meta(sid)

            mat = alpha_x[sid, rep_frame]
            if pred_label is None:
                title = f'Attention weights'
            else:
                title = f'Attention weights'

            save_heatmap(
                mat,
                title=title,
                save_path=os.path.join(save_dir, f'class_{cid}_sample_{sid}_representative_frame_attn.pdf'),
                cmap='viridis',
                xticklabels=joint_names,
                yticklabels=joint_names,
                highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
                highlight_color=HEATMAP_HIGHLIGHT_COLOR,
                highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
                exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
            )

            mean_mat = compute_mean_attention_from_alpha(alpha_x[sid])
            if pred_label is None:
                title = f'Class {cid} | Sample {sid} | True {true_label} | Mean Attention Over Time'
            else:
                title = f'Class {cid} | Sample {sid} | True {true_label} | Pred {pred_label} | Correct={is_correct} | Mean Attention Over Time'

            save_heatmap(
                mean_mat,
                title=title,
                save_path=os.path.join(save_dir, f'class_{cid}_sample_{sid}_mean_time_attn.pdf'),
                cmap='magma',
                xticklabels=joint_names,
                yticklabels=joint_names,
                highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
                highlight_color=HEATMAP_HIGHLIGHT_COLOR,
                highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
                exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
            )

            node_imp = compute_node_importance_from_alpha(alpha_x[sid])

            id_to_label = {
                    0: 'eating',
                    1: 'refusal food',
                    2: 'intention to eat',
                    3: 'waiting to eat',
                    4: 'social intention'
                }
            if pred_label is None:
                # title = f'Class {cid} | True {true_label} | Node Importance'
                title = f'Node Importance'
            else:
                # title = f'Class {cid} | True {true_label} | Pred {pred_label} | Correct={is_correct} | Node Importance'
                title = f'Node Importance for a Correctly Classified Sample in {id_to_label[cid]}'

            save_bar(
                node_imp,
                title=title,
                save_path=os.path.join(save_dir, f'class_{cid}_sample_{sid}_node_importance.pdf'),
                ylim=global_ylim,
                topk=topk,
                annotate_topk=True,
                highlight_topk=True,
                xtick_labels=joint_names
            )

            print(f'Saved individual plots for class {cid}, sample {sid}, representative frame {rep_frame}')
            print_topk(node_imp, f'Class {cid} Sample {sid}', topk=topk)


# =========================
# 7. 全局平均 attention
# =========================
def plot_global_mean_attention():
    mat = compute_mean_attention_from_alpha(alpha_x)
    title = 'Global Mean Attention'
    save_path = os.path.join(save_dir, 'global_mean_attention.pdf')
    save_heatmap(
        mat, title, save_path, cmap='coolwarm',
        xticklabels=joint_names, yticklabels=joint_names,
        highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
        highlight_color=HEATMAP_HIGHLIGHT_COLOR,
        highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
        exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
    )
    print(f'Saved: {save_path}')


def plot_global_node_importance(global_ylim=None, topk=10):
    node_imp = compute_node_importance_from_alpha(alpha_x)
    title = 'Global Node Importance'
    save_path = os.path.join(save_dir, 'global_node_importance.pdf')

    save_bar(
        node_imp, title, save_path,
        ylim=global_ylim, topk=topk,
        annotate_topk=True, highlight_topk=True,
        xtick_labels=joint_names
    )
    print(f'Saved: {save_path}')
    print_topk(node_imp, 'Global', topk=topk)


# =========================
# 8. 按 label 聚合
# =========================
def plot_class_mean_attention(class_id):
    cls_idx = np.where(labels == class_id)[0]
    if len(cls_idx) == 0:
        print(f'Class {class_id} has no samples.')
        return

    cls_mean = compute_mean_attention_from_alpha(alpha_x[cls_idx])

    title = f'Class {class_id} Mean Attention'
    save_path = os.path.join(save_dir, f'class_{class_id}_mean_attention.pdf')
    save_heatmap(
        cls_mean, title, save_path, cmap='viridis',
        xticklabels=joint_names, yticklabels=joint_names,
        highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
        highlight_color=HEATMAP_HIGHLIGHT_COLOR,
        highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
        exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
    )
    print(f'Saved: {save_path}, num_samples={len(cls_idx)}')


def plot_class_node_importance(class_id, global_ylim=None, topk=10):
    cls_idx = np.where(labels == class_id)[0]
    if len(cls_idx) == 0:
        print(f'Class {class_id} has no samples.')
        return

    node_imp = compute_node_importance_from_alpha(alpha_x[cls_idx])

    # title = f'Class {class_id} Node Importance'
    title = f'Node Importance'
    save_path = os.path.join(save_dir, f'class_{class_id}_node_importance.pdf')

    save_bar(
        node_imp, title, save_path,
        ylim=global_ylim, topk=topk,
        annotate_topk=True, highlight_topk=True,
        xtick_labels=joint_names
    )

    print(f'Saved: {save_path}, num_samples={len(cls_idx)}')
    print(f'Class {class_id}: min={node_imp.min():.6f}, max={node_imp.max():.6f}, std={node_imp.std():.6f}, range={(node_imp.max() - node_imp.min()):.6f}')
    print_topk(node_imp, f'Class {class_id}', topk=topk)


def plot_all_classes(topk=10, global_ylim=None):
    class_ids = get_class_ids()
    print('Unique labels:', class_ids)

    for cid in class_ids:
        plot_class_mean_attention(cid)
        plot_class_node_importance(cid, global_ylim=global_ylim, topk=topk)


# =========================
# 9. 类别差异图
# =========================
def plot_class_diff_attention(class_a, class_b):
    idx_a = np.where(labels == class_a)[0]
    idx_b = np.where(labels == class_b)[0]

    if len(idx_a) == 0 or len(idx_b) == 0:
        print(f'Class {class_a} or {class_b} has no samples.')
        return

    mean_a = compute_mean_attention_from_alpha(alpha_x[idx_a])
    mean_b = compute_mean_attention_from_alpha(alpha_x[idx_b])
    diff = mean_a - mean_b

    title = f'Attention Difference: Class {class_a} - Class {class_b}'
    save_path = os.path.join(save_dir, f'class_{class_a}_minus_class_{class_b}_attention.pdf')
    save_heatmap(
        diff, title, save_path, cmap='bwr', center=0,
        xticklabels=joint_names, yticklabels=joint_names,
        highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
        highlight_color=HEATMAP_HIGHLIGHT_COLOR,
        highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
        exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
    )
    print(f'Saved: {save_path}')


# =========================
# 10. 正确/错误分类分组可视化
# =========================
def plot_correct_wrong_mean_attention(global_ylim=None, topk=10):
    if y_pred is None:
        print('No prediction results found.')
        return

    correct_mask = (y_true == y_pred)
    wrong_mask = (y_true != y_pred)

    if correct_mask.sum() > 0:
        correct_mean = compute_mean_attention_from_alpha(alpha_x[correct_mask])
        save_heatmap(
            correct_mean,
            'Correctly Classified Mean Attention',
            os.path.join(save_dir, 'correct_mean_attention.pdf'),
            cmap='Greens',
            xticklabels=joint_names,
            yticklabels=joint_names,
            highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
            highlight_color=HEATMAP_HIGHLIGHT_COLOR,
            highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
            exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
        )
        print('Saved: correct_mean_attention.pdf')

        correct_imp = compute_node_importance_from_alpha(alpha_x[correct_mask])
        save_bar(
            correct_imp,
            'Correctly Classified Node Importance',
            os.path.join(save_dir, 'correct_node_importance.pdf'),
            ylim=global_ylim,
            topk=topk,
            annotate_topk=True,
            highlight_topk=True,
            xtick_labels=joint_names
        )
        print('Saved: correct_node_importance.pdf')
        print_topk(correct_imp, 'Correct samples', topk=topk)

    if wrong_mask.sum() > 0:
        wrong_mean = compute_mean_attention_from_alpha(alpha_x[wrong_mask])
        save_heatmap(
            wrong_mean,
            'Misclassified Mean Attention',
            os.path.join(save_dir, 'wrong_mean_attention.pdf'),
            cmap='Reds',
            xticklabels=joint_names,
            yticklabels=joint_names,
            highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
            highlight_color=HEATMAP_HIGHLIGHT_COLOR,
            highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
            exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
        )
        print('Saved: wrong_mean_attention.pdf')

        wrong_imp = compute_node_importance_from_alpha(alpha_x[wrong_mask])
        save_bar(
            wrong_imp,
            'Misclassified Node Importance',
            os.path.join(save_dir, 'wrong_node_importance.pdf'),
            ylim=global_ylim,
            topk=topk,
            annotate_topk=True,
            highlight_topk=True,
            xtick_labels=joint_names
        )
        print('Saved: wrong_node_importance.pdf')
        print_topk(wrong_imp, 'Wrong samples', topk=topk)

    if correct_mask.sum() > 0 and wrong_mask.sum() > 0:
        diff = compute_mean_attention_from_alpha(alpha_x[correct_mask]) - compute_mean_attention_from_alpha(alpha_x[wrong_mask])
        save_heatmap(
            diff,
            'Attention Difference: Correct - Wrong',
            os.path.join(save_dir, 'correct_minus_wrong_attention.pdf'),
            cmap='bwr',
            center=0,
            xticklabels=joint_names,
            yticklabels=joint_names,
            highlight_topk=HEATMAP_HIGHLIGHT_TOPK,
            highlight_color=HEATMAP_HIGHLIGHT_COLOR,
            highlight_linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
            exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL
        )
        print('Saved: correct_minus_wrong_attention.pdf')


# =========================
# 11. 类别node importance叠加比较图
# =========================
def plot_all_class_node_importance_in_one():
    class_ids = get_class_ids()
    fig, ax = plt.subplots(figsize=(12, 5))

    for cid in class_ids:
        cls_idx = np.where(labels == cid)[0]
        if len(cls_idx) == 0:
            continue
        node_imp = compute_node_importance_from_alpha(alpha_x[cls_idx])
        ax.plot(
            np.arange(len(node_imp)),
            node_imp,
            marker='o',
            markersize=4,
            linewidth=1.5,
            label=f'Class {cid}'
        )

    ax.set_xticks(np.arange(V))
    ax.set_xticklabels(joint_names, fontname=FONT_NAME, fontsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    style_axis_text(
        ax,
        title='Node Importance Comparison Across Classes',
        xlabel='Landmark',
        ylabel='Importance'
    )

    leg = ax.legend(frameon=True, fontsize=LEGEND_SIZE)
    for text in leg.get_texts():
        text.set_fontname(FONT_NAME)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'all_classes_node_importance_comparison.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {save_path}')


# =========================
# 12. 类别统计与类间距离
# =========================
def save_class_node_importance_stats():
    class_ids = get_class_ids()
    class_imp_dict = get_class_node_importance_dict()

    stats_path = os.path.join(save_dir, 'class_node_importance_stats.txt')
    with open(stats_path, 'w') as f:
        f.write('Class Node Importance Statistics\n')
        f.write('=================================\n\n')

        for cid in class_ids:
            imp = class_imp_dict[cid]
            f.write(f'Class {cid}\n')
            f.write(f'  min   : {imp.min():.6f}\n')
            f.write(f'  max   : {imp.max():.6f}\n')
            f.write(f'  mean  : {imp.mean():.6f}\n')
            f.write(f'  std   : {imp.std():.6f}\n')
            f.write(f'  range : {(imp.max() - imp.min()):.6f}\n')

            top_idx = np.argsort(imp)[-10:][::-1]
            f.write(f'  top10 :\n')
            for rank, idx in enumerate(top_idx, start=1):
                f.write(f'    #{rank}: node {idx} ({joint_names[idx]}), importance={imp[idx]:.6f}\n')
            f.write('\n')

    print(f'Saved: {stats_path}')


def save_class_pairwise_distance():
    class_ids = get_class_ids()
    class_imp_dict = get_class_node_importance_dict()

    dist_mat = np.zeros((len(class_ids), len(class_ids)), dtype=np.float32)

    for i, ci in enumerate(class_ids):
        for j, cj in enumerate(class_ids):
            vi = class_imp_dict[ci]
            vj = class_imp_dict[cj]
            dist_mat[i, j] = np.linalg.norm(vi - vj)

    npy_path = os.path.join(save_dir, 'class_node_importance_pairwise_distance.npy')
    np.save(npy_path, dist_mat)

    txt_path = os.path.join(save_dir, 'class_node_importance_pairwise_distance.txt')
    with open(txt_path, 'w') as f:
        f.write('Pairwise L2 Distance of Class Node Importance\n')
        f.write('============================================\n\n')

        header = 'class\t' + '\t'.join([str(c) for c in class_ids]) + '\n'
        f.write(header)
        for i, ci in enumerate(class_ids):
            row = [f'{dist_mat[i, j]:.6f}' for j in range(len(class_ids))]
            f.write(f'{ci}\t' + '\t'.join(row) + '\n')

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        dist_mat,
        ax=ax,
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        xticklabels=class_ids,
        yticklabels=class_ids,
        annot_kws={"fontsize": ANNOT_SIZE, "fontname": FONT_NAME}
    )

    style_axis_text(
        ax,
        title='Pairwise L2 Distance of Class Node Importance',
        xlabel='Class',
        ylabel='Class'
    )

    fig_path = os.path.join(save_dir, 'class_node_importance_pairwise_distance_heatmap.pdf')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {npy_path}')
    print(f'Saved: {txt_path}')
    print(f'Saved: {fig_path}')


# =========================
# 13. 论文风格大图
# =========================
def plot_paper_style_selected_samples_panel(
    class_ids=(0, 1),
    num_samples_per_class=2,
    topk=10,
    global_ylim=None,
    prefer_correct=True,
    frame_select_method='offdiag_sum'
):
    selected_info = []

    for cid in class_ids:
        selected_samples = select_samples_for_class(
            class_id=cid,
            num_samples=num_samples_per_class,
            prefer_correct=prefer_correct
        )

        for sid in selected_samples:
            rep_frame = get_representative_frame_id(alpha_x[sid], method=frame_select_method)
            selected_info.append((cid, sid, rep_frame))

    if len(selected_info) == 0:
        print('No samples selected.')
        return

    nrows = len(selected_info)
    ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    all_frame_mats = []
    all_mean_mats = []

    for cid, sid, rep_frame in selected_info:
        all_frame_mats.append(alpha_x[sid, rep_frame])
        all_mean_mats.append(compute_mean_attention_from_alpha(alpha_x[sid]))

    frame_vmin = min([m.min() for m in all_frame_mats])
    frame_vmax = max([m.max() for m in all_frame_mats])

    mean_vmin = min([m.min() for m in all_mean_mats])
    mean_vmax = max([m.max() for m in all_mean_mats])

    for row, (cid, sid, rep_frame) in enumerate(selected_info):
        frame_mat = alpha_x[sid, rep_frame]
        mean_mat = compute_mean_attention_from_alpha(alpha_x[sid])
        node_imp = compute_node_importance_from_alpha(alpha_x[sid])
        true_label, pred_label, is_correct = get_sample_meta(sid)

        if pred_label is None:
            row_prefix = f'Class {cid} | Sample {sid} | True {true_label} | Frame {rep_frame}'
        else:
            row_prefix = f'Class {cid} | Sample {sid} | True {true_label} | Pred {pred_label} | Correct={is_correct} | Frame {rep_frame}'

        # col 1
        ax = axes[row, 0]
        sns.heatmap(
            frame_mat,
            ax=ax,
            cmap='viridis',
            vmin=frame_vmin,
            vmax=frame_vmax,
            cbar=True,
            xticklabels=joint_names if row == nrows - 1 else False,
            yticklabels=joint_names
        )
        draw_topk_rectangles(
            ax=ax,
            mat=frame_mat,
            topk=HEATMAP_HIGHLIGHT_TOPK,
            color=HEATMAP_HIGHLIGHT_COLOR,
            linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
            exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL,
            verbose=False,
            title=f'{row_prefix} Representative Frame Attention'
        )
        style_axis_text(
            ax,
            title='Representative Frame Attention',
            xlabel='Target landmarks' if row == nrows - 1 else '',
            ylabel='Source landmarks'
        )

        # col 2
        ax = axes[row, 1]
        sns.heatmap(
            mean_mat,
            ax=ax,
            cmap='magma',
            vmin=mean_vmin,
            vmax=mean_vmax,
            cbar=True,
            xticklabels=joint_names if row == nrows - 1 else False,
            yticklabels=False
        )
        draw_topk_rectangles(
            ax=ax,
            mat=mean_mat,
            topk=HEATMAP_HIGHLIGHT_TOPK,
            color=HEATMAP_HIGHLIGHT_COLOR,
            linewidth=HEATMAP_HIGHLIGHT_LINEWIDTH,
            exclude_diagonal=HEATMAP_EXCLUDE_DIAGONAL,
            verbose=False,
            title=f'{row_prefix} Mean Attention Over Time'
        )
        style_axis_text(
            ax,
            title='Mean Attention Over Time',
            xlabel='Target landmarks' if row == nrows - 1 else '',
            ylabel=''
        )

        # col 3
        ax = axes[row, 2]
        x = np.arange(len(node_imp))
        topk_ = min(topk, len(node_imp))
        top_idx = np.argsort(node_imp)[-topk_:][::-1]
        top_set = set(top_idx.tolist())

        colors = ['tomato' if i in top_set else 'steelblue' for i in range(len(node_imp))]
        ax.bar(x, node_imp, color=colors, edgecolor='black', linewidth=0.5)

        if global_ylim is not None:
            ax.set_ylim(global_ylim)

        ymax = global_ylim[1] if global_ylim is not None else node_imp.max()
        text_offset = ymax * 0.01 if ymax > 0 else 0.001

        for rank, i in enumerate(top_idx, start=1):
            v = node_imp[i]
            ax.text(
                i, v + text_offset,
                f'#{rank}\n{i}',
                ha='center', va='bottom',
                fontsize=ANNOT_SIZE,
                rotation=90,
                fontname=FONT_NAME
            )

        ax.set_xticks(x)
        if row == nrows - 1:
            ax.set_xticklabels(joint_names, rotation=0, fontname=FONT_NAME, fontsize=TICK_SIZE)
        else:
            ax.set_xticklabels([])

        style_axis_text(
            ax,
            title='Node Importance',
            xlabel='Landmark' if row == nrows - 1 else '',
            ylabel='Importance'
        )

        # 行左侧增加样本信息，更适合论文阅读
        ax0 = axes[row, 0]
        ax0.text(
            -0.62, 0.5,
            row_prefix,
            transform=ax0.transAxes,
            rotation=90,
            va='center',
            ha='center',
            fontsize=10,
            fontname=FONT_NAME
        )

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'paper_style_class0_class1_selected_samples_panel.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Saved: {save_path}')


# =========================
# 14. 主程序
# =========================
if __name__ == '__main__':
    topk = 11

    unified_ylim = get_unified_node_importance_ylim(top_margin=1.15)
    print(f'\nUnified y-axis for all node importance plots: {unified_ylim}')

    class_ids, class_counts = np.unique(labels, return_counts=True)
    print('\nLabel distribution:')
    for cid, cnt in zip(class_ids, class_counts):
        print(f'  Class {cid}: {cnt} samples')

    # A
    plot_selected_samples_individually(
        class_ids=(0, 1),
        num_samples_per_class=2,
        topk=topk,
        global_ylim=None,
        prefer_correct=True,
        frame_select_method='offdiag_sum'
    )

    # B
    plot_paper_style_selected_samples_panel(
        class_ids=(0, 1),
        num_samples_per_class=2,
        topk=topk,
        global_ylim=None,
        prefer_correct=True,
        frame_select_method='offdiag_sum'
    )

    # C
    plot_global_mean_attention()
    plot_global_node_importance(global_ylim=None, topk=topk)

    plot_all_classes(topk=topk, global_ylim=None)
    plot_all_class_node_importance_in_one()

    class_ids = get_class_ids()
    if len(class_ids) >= 2:
        plot_class_diff_attention(class_ids[0], class_ids[1])

    plot_correct_wrong_mean_attention(global_ylim=None, topk=topk)

    save_class_node_importance_stats()
    save_class_pairwise_distance()

    print(f'\nAll figures are saved to: {save_dir}')
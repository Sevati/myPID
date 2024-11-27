import os
# 禁用 Intel MKL 警告
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ["MKL_ENABLE_SSE_DEPRECATION_WARNINGS"] = "0"

# 忽略所有警告
import warnings
warnings.filterwarnings('ignore', message="Intel MKL WARNING")
warnings.filterwarnings('ignore')


import numpy as np
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from LoadData import load_data, get_hits, processing_data, reassign_labels
import scipy
import pandas as pd

from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from collections import defaultdict

EvtNumTrain = 1000
file_path = "/Users/Sevati/PycharmProjects/untitled/PID/pid_data/MCTdata/hit_2.txt"
df_train = load_data(file_path)

def filter_rows(df, min_points = 5):
    # 计算每个类别的点数量
    label_counts = df['trkid'].value_counts()

    # 只有点数量>5的类别才被保留
    valid_labels = label_counts[label_counts > min_points].index

    # 过滤DataFrame，只保留valid_labels中的类别行
    filtered_df = df[df['trkid'].isin(valid_labels)]

    return filtered_df


# 自定义的损失函数，用于评价聚类效果
def calculate_efficiency_and_purity(data, gt_labels, method_labels):
    def get_indices(labels):
        indices = defaultdict(list)
        for index, label in enumerate(labels):
            indices[label].append(index)
        return indices
    unique_labels = np.unique(gt_labels)  # 包括所有标签，不过滤噪声
    
    # 获得每个类别的索引
    true_indices = get_indices(gt_labels)
    cluster_indices = get_indices(method_labels)

    efficiencies = {}
    purities = {}

    # 建立真值类别和聚类结果类别之间的对应关系
    true_to_cluster_map = {}

    # 计算真值中的每个类别在聚类结果中的最大匹配类别
    for true_class, gt_indices in true_indices.items():
        cluster_count = defaultdict(int)
        for idx in gt_indices:
            cluster_count[method_labels[idx]] += 1
        mapped_cluster = max(cluster_count, key=cluster_count.get)
        true_to_cluster_map[true_class] = mapped_cluster

    # 计算每个类别的效率和纯度
    for true_class, mapped_cluster in true_to_cluster_map.items():
        gt_indices = true_indices[true_class]
        db_indices = cluster_indices[mapped_cluster]
        
        correct_count_efficiency = sum(1 for idx in gt_indices if method_labels[idx] == mapped_cluster)
        correct_count_purity = sum(1 for idx in db_indices if gt_labels[idx] == true_class)
        
        efficiency = correct_count_efficiency / len(gt_indices) if len(gt_indices) > 0 else 0
        purity = correct_count_purity / len(db_indices) if len(db_indices) > 0 else 0
        
        for idx in gt_indices:
            efficiencies[idx] = efficiency
            purities[idx] = purity
    
    data['efficiency'] = efficiencies
    data['purity'] = purities
    return 


def calculate_success_rate(data, gt_labels, method_labels, successful_classes, total_classes):
    # 计算效率和纯度
    calculate_efficiency_and_purity(data, gt_labels, method_labels)
    total_classes += len(np.unique(gt_labels))  # 包括所有真实类（噪声不算）

    cutdata = filter_rows(data, min_points=5)
    df = pd.DataFrame(cutdata)

    unique_data = df.drop_duplicates(subset=['trkid'])
    unique_data = unique_data.reset_index(drop=True)

    # print(unique_data)

    for k in cutdata['trkid'].unique():
        efficiency = unique_data[unique_data['trkid'] == k]['efficiency'].values[0]
        purity = unique_data[unique_data['trkid'] == k]['purity'].values[0]
        if efficiency > 0.8 and purity > 0.6:
            successful_classes += 1

    return successful_classes, total_classes


def custom_loss(hit, gt_labels_list, pred_labels_list):
    successful_classes = 0
    total_classes = 0    
    successful_classes, total_classes = calculate_success_rate(hit, gt_labels_list, pred_labels_list, successful_classes, total_classes)
    success_rate = successful_classes / total_classes if total_classes > 0 else 0
    L_max = 100
    return L_max * (1 - success_rate)  # 成功率越大，loss越小


# 定义目标函数，计算DBSCAN的损失
def dbscan_objective(params, train_data, train_labels):
    # print(f'params: {params}')
    eps, min_samples = params
    loss = {}

    for evtCount in train_data:
        X = train_data[evtCount]
        if X.shape[0] < 10:
            continue
        hit =  get_hits(df_train, evtCount)
        clustering = DBSCAN(eps=eps, min_samples=int(min_samples))
        db = clustering.fit(X)
        pred_labels = db.labels_
        pred_labels = reassign_labels(pred_labels)
        hit['cluster'] = pred_labels
        # all_pred_labels.append(pred_labels)

        y = train_labels[evtCount]
        y = reassign_labels(y)
        # all_truth_labels.append(y)

        loss[evtCount] = custom_loss(hit, y, pred_labels)
        # print(f'evtCount: {evtCount}, loss: {loss[evtCount]}')
    
    average_loss = np.mean(list(loss.values()))  # 计算所有事件的平均损失
    return average_loss


# 定义 MeanShift 的目标函数，计算损失
def meanshift_objective(params, train_data, train_labels):

    quantile, n_samples = params
    loss = {}
    
    for evtCount in train_data:
        # print(f'evtCount: {evtCount}')
        X = train_data[evtCount]
        if X.shape[0] < 10:
            continue
        hit =  get_hits(df_train, evtCount)
        bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=int(n_samples))
        # 如果 bandwidth 为 0.0，则重新估算
        bandwidth = max(bandwidth, 0.1)
        # bandwidth = estimate_bandwidth(X, quantile=min(quantile+0.1, 1.0), n_samples=min(int(n_samples*1.5), len(X)))
        clustering = MeanShift(bandwidth=bandwidth)
        ms = clustering.fit(X)
        pred_labels = ms.labels_
        pred_labels = reassign_labels(pred_labels)
        hit['cluster'] = pred_labels
        y = train_labels[evtCount]
        y = reassign_labels(y)

        loss[evtCount] = custom_loss(hit, y, pred_labels)
    
    average_loss = np.mean(list(loss.values()))  # 计算所有事件的平均损失
    return average_loss


if __name__ == '__main__':
    train_data = {}
    train_labels = {}
    for evtCount in range(EvtNumTrain):
        hits = get_hits(df_train, evtCount)
        train_data[evtCount] = hits[['finalX', 'finalY']].values
        train_labels[evtCount] = hits['trkid'].values

    # train_data, test_data, train_labels, test_labels = processing_data(file_path)


    # 用户选择要优化的算法
    algorithm_choice = 'meanshift'  # 或者 'meanshift'

    if algorithm_choice == 'dbscan':
        # 定义 DBSCAN 参数空间
        space = [
            Real(0.1, 10.0, name='eps'),
            Integer(1, 50, name='min_samples')
        ]
        objective_function = dbscan_objective

    elif algorithm_choice == 'meanshift':
       # 定义 MeanShift 参数空间
        space = [
            Real(0.1, 0.9, name='quantile'),
            Integer(1, 500, name='n_samples')
        ]
        objective_function = meanshift_objective


    # 使用贝叶斯优化来最小化目标函数
    result = gp_minimize(lambda params: objective_function(params, train_data, train_labels), space, n_calls=50, n_random_starts=10, random_state=42)

    # 输出最佳参数
    if algorithm_choice == 'dbscan':
        best_eps = result.x[0]
        best_min_samples = result.x[1]
        print(f'最佳参数: eps={best_eps}, min_samples={best_min_samples}')
    elif algorithm_choice == 'meanshift':
        best_quantile = result.x[0]
        best_n_samples = result.x[1]
        print(f'最佳参数: quantile={best_quantile}, n_samples={best_n_samples}')

    print("最小化目标函数值: {}".format(result.fun))


    # 输出每次迭代的损失历史
    loss_history = result.func_vals
    print(f'损失历史: {loss_history}')


    #  # 利用找到的最佳参数
    # if algorithm_choice == 'dbscan':
    #     best_clustering = DBSCAN(eps=best_eps, min_samples=int(best_min_samples))
    # elif algorithm_choice == 'meanshift':
    #     # 使用最佳参数估算带宽
    #     X_sample = train_data[0]  # 使用第一个数据集作为示例来估算带宽
    #     best_bandwidth = estimate_bandwidth(X_sample, quantile=best_quantile, n_samples=int(best_n_samples))
    #     best_clustering = MeanShift(bandwidth=best_bandwidth)
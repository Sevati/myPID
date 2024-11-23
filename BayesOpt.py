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
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


# 自定义的损失函数，用于评价聚类效果
def calculate_efficiency_and_purity(gt_labels, dbscan_labels):
    unique_labels = np.unique(gt_labels)  # 包括所有标签，不过滤噪声
    results = {}

    for label in unique_labels:
        # ground truth 中该类的所有索引
        gt_indices = np.where(gt_labels == label)[0]
        db_indices = np.where(dbscan_labels == label)[0]

        # 计算与 ground truth 标签一致的点数
        correct_count = np.sum(np.in1d(db_indices, gt_indices))

        # 计算该类的效率
        efficiency = correct_count / len(gt_indices) if len(gt_indices) > 0 else 0

        # 计算纯度
        purity = correct_count / len(db_indices) if len(db_indices) > 0 else 0

        results[label] = {'efficiency': efficiency, 'purity': purity}

    return results


def calculate_success_rate(gt_labels, method_labels, successful_classes, total_classes):
    # 计算效率和纯度
    results = calculate_efficiency_and_purity(gt_labels, method_labels)
    total_classes += len(np.unique(gt_labels))  # 包括所有真实类（噪声也算）

    for label, metrics in results.items():
        efficiency = metrics['efficiency']
        purity = metrics['purity']

        if efficiency > 0.8 and purity > 0.6:
            successful_classes += 1

    return successful_classes, total_classes


def custom_loss(gt_labels_list, pred_labels_list):
    successful_classes = 0
    total_classes = 0
    for gt_labels, pred_labels in zip(gt_labels_list, pred_labels_list):
        successful_classes, total_classes = calculate_success_rate(gt_labels, pred_labels, successful_classes, total_classes)
    success_rate = successful_classes / total_classes if total_classes > 0 else 0
    L_max = 100
    return L_max * (1 - success_rate)  # 成功率越大，loss越小


# 定义目标函数，计算DBSCAN的损失
def dbscan_objective(params, train_data, train_labels):
    eps, min_samples = params
    loss = {}

    for evtCount in train_data:
        X = train_data[evtCount]
        if X.shape[0] < 10:
            continue
        
        clustering = DBSCAN(eps=eps, min_samples=int(min_samples))
        db = clustering.fit(X)
        pred_labels = db.labels_
        pred_labels = reassign_labels(pred_labels)
        # all_pred_labels.append(pred_labels)

        y = train_labels[evtCount]
        y = reassign_labels(y)
        # all_truth_labels.append(y)

        loss[evtCount] = custom_loss(y, pred_labels)
    
    average_loss = np.mean(list(loss.values()))  # 计算所有事件的平均损失
    return average_loss


# 定义 MeanShift 的目标函数，计算损失
def meanshift_objective(params, train_data, train_labels):

    quantile, n_samples = params
    loss = {}
    
    for evtCount in train_data:
        X = train_data[evtCount]
        if X.shape[0] < 10:
            continue
        
        bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=int(n_samples))
        clustering = MeanShift(bandwidth=bandwidth)
        ms = clustering.fit(X)
        pred_labels = ms.labels_
        pred_labels = reassign_labels(pred_labels)

        y = train_labels[evtCount]
        y = reassign_labels(y)

        loss[evtCount] = custom_loss(y, pred_labels)
    
    average_loss = np.mean(list(loss.values()))  # 计算所有事件的平均损失
    return average_loss


if __name__ == '__main__':
    EvtNumTrain = 1000
    file_path = "/Users/Sevati/PycharmProjects/untitled/PID/pid_data/MCTdata/hit_2.txt"
    df_train = load_data(file_path)
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
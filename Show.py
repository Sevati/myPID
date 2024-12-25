import matplotlib.cm as cm
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
from collections import defaultdict

from LoadData import load_data, get_hits, reassign_labels
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
# from BayesOpt import calculate_success_rate


def dbscan_clustering(data, eps, min_samples=4):
    # Extract x and y coordinates
    X = data[['finalX', 'finalY']].values

    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    # labels = reassign_labels(labels)

    data['dbscan_label'] = labels

    # Number of clusters in labels, ignoring noise (-1 label)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return data, n_clusters_


def meanshift_clustering(data, quantile=0.5, n_samples=100):
    # Extract x and y coordinates
    X = data[['finalX', 'finalY']].values
    # Estimate bandwidth
    bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)
    # Apply Mean Shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    # labels = reassign_labels(labels)

    # Add the labels back to the data to keep the original point order
    data['meanshift_label'] = labels

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print('Estimated number of clusters (Mean Shift): %d' % n_clusters_)
    return labels, n_clusters_


#原始版本计算效率和纯度
def calculate_efficiency_and_purity(data, label_name, gt_labels, cluster_labels, efficiency_list, purity_list):
    unique_labels = np.unique(gt_labels)  # 包括所有标签，不过滤噪声
    results = {}
    successful_classes = 0

    for label in unique_labels:
        # ground truth 中该类的所有索引
        gt_indices = np.where(gt_labels == label)[0]
        db_indices = np.where(cluster_labels == label)[0]

        # 计算与 ground truth 标签一致的点数
        correct_count = np.sum(np.in1d(db_indices, gt_indices))

        # 计算该类的效率
        efficiency = correct_count / len(gt_indices) if len(gt_indices) > 0 else 0
        
        # 计算纯度
        purity = correct_count / len(db_indices) if len(db_indices) > 0 else 0

        data[f'{label_name}_efficiency'] = efficiency
        data[f'{label_name}_purity'] = purity
        
        efficiency_list.append(efficiency)
        purity_list.append(purity)
        
        # 判断寻类是否成功
        if efficiency > 0.8 and purity > 0.6:
            successful_classes += 1

    # 计算寻类成功率
    success_rate = successful_classes / len(unique_labels) if len(unique_labels) > 0 else 0

    return 



def filter_rows(df, min_points = 5):
    # 计算每个类别的点数量
    label_counts = df['trkid'].value_counts()

    # 只有点数量>=3的类别才被保留
    valid_labels = label_counts[label_counts > min_points].index

    # 过滤DataFrame，只保留valid_labels中的类别行
    filtered_df = df[df['trkid'].isin(valid_labels)]

    return filtered_df


#新版本计算效率和纯度
def compute_efficiency_and_purity(data, label_name, gt_labels, method_labels, efficiency_list, purity_list):
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
        
        # data[f'{label_name}_efficiency'] = efficiency
        # data[f'{label_name}_purity'] = purity
        
        efficiency_list.append(efficiency)
        purity_list.append(purity)

        for idx in gt_indices:
            efficiencies[idx] = efficiency
            purities[idx] = purity
    
    data[f'{label_name}_efficiency'] = efficiencies
    data[f'{label_name}_purity'] = purities
        
    return 


def calculate_success_rate(data, label_name, gt_labels, method_labels, efficiency_list, purity_list, successful_classes, total_classes):
    # 计算效率和纯度
    compute_efficiency_and_purity(data, label_name, gt_labels, method_labels, efficiency_list, purity_list)
    total_classes += len(np.unique(gt_labels))  # 包括所有真实类（噪声也算）

    cutdata = filter_rows(data, min_points=5)
    df = pd.DataFrame(cutdata)

    unique_data = df.drop_duplicates(subset=['trkid'])
    unique_data = unique_data.reset_index(drop=True)
    for k in cutdata['trkid'].unique():
        efficiency = unique_data[unique_data['trkid'] == k][f'{label_name}_efficiency'].values[0]
        purity = unique_data[unique_data['trkid'] == k][f'{label_name}_purity'].values[0]
        if efficiency > 0.8 and purity > 0.6:
            successful_classes += 1

    return successful_classes, total_classes, unique_data


def evalute_clusters(data, evtCount, efficiency_list, purity_list, successful_classes, total_classes, label_name):
    unique_trkids = data['trkid'].unique()
    trkid_colors = cm.rainbow(np.linspace(0, 1, len(unique_trkids)))
    gt_labels = data['trkid'].values

    cluster_labels = data[label_name].values
    
    #标签对齐以计算效率和纯度
    gt_LABLES = reassign_labels(gt_labels)
    cluster_LABLES = reassign_labels(cluster_labels)
    
    #计算成功率，输入为聚类标签和真实标签，successful_classes 和 total_classes 用来记录总径迹数和成功寻迹总数
    successful_classes, total_classes, unique_data = calculate_success_rate(data, label_name, gt_LABLES, cluster_LABLES, efficiency_list, purity_list, successful_classes, total_classes)
    return successful_classes, total_classes, unique_data 


def visualize_clusters(data, unique_data):
    # df = pd.DataFrame(data)
    # # 删除重复的trkid，仅保留第一次出现的记录
    # unique_data = df.drop_duplicates(subset=['trkid'])
    # # 重置索引
    # unique_data = unique_data.reset_index(drop=True)
    
    unique_trkids = data['trkid'].unique()
    unique_dbscan_labels = data['dbscan_label'].unique()
    unique_meanshift_labels = data['meanshift_label'].unique()

    trkid_colors = cm.rainbow(np.linspace(0, 1, len(unique_trkids)))

    #显示结果子图2*3
    fig, axs = plt.subplots(3, 2, figsize=(20, 30))  # Adjust figsize to make the figure smaller

# '''
    # Subplot 1: Groundtruth labels (x, y)
    handles = []
    for k, col in zip(unique_trkids, trkid_colors):
        class_member_mask = (data['trkid'] == k)
        xy = data[class_member_mask]
        handle = axs[0, 0].scatter(xy['x'], xy['y'], s=30, color=col, label=f'Cluster {k} ({np.sum(data["trkid"] == k)})',edgecolor='k')
        handles.append(handle)
    axs[0, 0].set_title('Groundtruth Labels (x, y) - Event ' + str(evtCount))
    axs[0, 0].legend(handles=handles, loc='upper right')
    axs[0, 0].set_aspect('equal')  # 设置坐标轴比例相等


    # Subplot 3: DBSCAN clustering results
    # unique_dbscan_labels = set(dbscan_labels)
    dbscan_colors = cm.rainbow(np.linspace(0, 1, len(unique_dbscan_labels)))
    handles = []
    for k, col in zip(unique_dbscan_labels, dbscan_colors):
        class_member_mask = (data['dbscan_label'] == k)
        xy = data[class_member_mask]
        handle = axs[1, 0].scatter(xy['x'], xy['y'], s=30, color=col, label=f'Cluster {k} ({np.sum(data["dbscan_label"] == k)})',edgecolor='k')
        handles.append(handle)
    # 对每个唯一的trkid展示Efficiency和Purity
    y_offset = 0.02  # 初始偏移量
    text_gap = 0.05  # 每个文本间的间距
    for k, col in zip(unique_data['trkid'], trkid_colors):
        db_efficiency_value = unique_data[unique_data['trkid'] == k]['dbscan_label_efficiency'].values[0]
        db_purity_value = unique_data[unique_data['trkid'] == k]['dbscan_label_purity'].values[0]
        info_text = f'{k}  Accuracy: {db_efficiency_value:.2f}\n    Purity: {db_purity_value:.2f}'
        # axs[1, 0].text(0.02, 0.02, info_text, transform=axs[1, 0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        axs[1, 0].text(0.02, y_offset, info_text, transform=axs[1, 0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.75), color=col)
        y_offset += text_gap  # 更新偏移量
    axs[1, 0].set_title('DBSCAN Clustering Results - Event ' + str(evtCount))
    axs[1, 0].legend(handles=handles, loc='upper right')
    axs[1, 0].set_aspect('equal')
   

    # Subplot 5: meanshift clustering results
    meanshift_colors = cm.rainbow(np.linspace(0, 1, len(unique_meanshift_labels)))
    handles = []
    for k, col in zip(unique_meanshift_labels, meanshift_colors):
        class_member_mask = (data['meanshift_label'] == k)
        xy = data[class_member_mask]
        handle = axs[2, 0].scatter(xy['x'], xy['y'], s=30, color=col, label=f'Cluster {k} ({np.sum(data["meanshift_label"] == k)})',edgecolor='k')
        handles.append(handle)
   # 对每个唯一的trkid展示Efficiency和Purity
    y_offset = 0.02  # 初始偏移量
    text_gap = 0.05  # 每个文本间的间距
    for k, col in zip(unique_data['trkid'], trkid_colors):
        ms_efficiency_value = unique_data[unique_data['trkid'] == k]['meanshift_label_efficiency'].values[0]
        ms_purity_value = unique_data[unique_data['trkid'] == k]['meanshift_label_purity'].values[0]
        info_text = f'{k}  Accuracy: {ms_efficiency_value:.2f}\n    Purity: {ms_purity_value:.2f}'
        # axs[2, 0].text(0.02, 0.02, info_text, transform=axs[2, 0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        axs[2, 0].text(0.02, y_offset, info_text, transform=axs[2, 0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.75), color=col)
        y_offset += text_gap  # 更新偏移量
    axs[2, 0].set_title('Mean Shift Clustering Results - Event ' + str(evtCount))
    axs[2, 0].legend(handles=handles, loc='upper right')
    axs[2, 0].set_aspect('equal')
   
    # Subplot 2: Groundtruth labels (finalX, finalY)
    handles = []
    for k, col in zip(unique_trkids, trkid_colors):
        class_member_mask = (data['trkid'] == k)
        xy = data[class_member_mask]
        handle = axs[0, 1].scatter(xy['finalX'], xy['finalY'], s=30, color=col, label=f'Cluster {k} ({np.sum(data["trkid"] == k)})',edgecolor='k')
        handles.append(handle)
    axs[0, 1].set_title('Groundtruth Labels (finalX, finalY) - Event ' + str(evtCount))
    axs[0, 1].legend(handles=handles, loc='upper right')
    axs[0, 1].set_aspect('equal')
    
    # Subplot 4: DBSCAN clustering results (finalX, finalY)
    dbscan_colors = cm.rainbow(np.linspace(0, 1, len(unique_dbscan_labels)))
    handles = []
    for k, col in zip(unique_dbscan_labels, dbscan_colors):
        class_member_mask = (data['dbscan_label'] == k)
        xy = data[class_member_mask]
        handle = axs[1, 1].scatter(xy['finalX'], xy['finalY'], s=30, color=col, label=f'Cluster {k} ({np.sum(data["dbscan_label"] == k)})',edgecolor='k')
        handles.append(handle)
    axs[1, 1].set_title('DBSCAN Clustering Results (finalX, finalY) - Event ' + str(evtCount))
    axs[1, 1].legend(handles=handles, loc='upper right')
    axs[1, 1].set_aspect('equal')
    
    # Subplot 6: Mean Shift clustering results (finalX, finalY)
    meanshift_colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_meanshift_labels)))
    handles = []
    for k, col in zip(unique_meanshift_labels, meanshift_colors):
        class_member_mask = (data['meanshift_label'] == k)
        xy = data[class_member_mask]
        handle = axs[2, 1].scatter(xy['finalX'], xy['finalY'], s=30 , color=col, label=f'Cluster {k} ({np.sum(data["meanshift_label"] == k)})',edgecolor='k')
        handles.append(handle)
    axs[2, 1].set_title('Mean Shift Clustering Results (finalX, finalY) - Event ' + str(evtCount))
    axs[2, 1].legend(handles=handles, loc='upper right')
    axs[2, 1].set_aspect('equal')

    axs[0, 0].axis('square')
    axs[1, 0].axis('square')
    axs[2, 0].axis('square')
    axs[0, 1].axis('square')
    axs[1, 1].axis('square')
    axs[2, 1].axis('square')
    
    plt.tight_layout()  # Adjust layout to fit the figure better
    plt.savefig('/Users/Sevati/PycharmProjects/untitled/PID/Axs_Results_2/axs/event' + str(evtCount) + '.jpg')
    plt.close()
    return


def pick_failure(data):
    cutdata = filter_rows(data, min_points=5)
    df = pd.DataFrame(cutdata)
    # 删除重复的trkid，仅保留第一次出现的记录
    unique_data = df.drop_duplicates(subset=['trkid'])
    # 重置索引
    unique_data = unique_data.reset_index(drop=True)

    db_efficiency = unique_data['dbscan_label_efficiency'].values
    db_purity = unique_data['dbscan_label_purity'].values
    ms_efficiency = unique_data['meanshift_label_efficiency'].values
    ms_purity = unique_data['meanshift_label_purity'].values

    # 判断是否有任一数值小于1 
    # any(db_efficiency < 0.8) or any(db_purity < 0.6) or 
    if any(db_efficiency < 0.8) or any(db_purity < 0.6):
        ifsave = True
    else:
        ifsave = False
    if ifsave:
            src_path = '/Users/Sevati/PycharmProjects/untitled/PID/Axs_Results_2/axs/event' + str(evtCount) + '.jpg'
            dst_path = '/Users/Sevati/PycharmProjects/untitled/PID/Axs_Results_2/axs_failure_db/event' + str(evtCount) + '.jpg'
            # 创建目标目录（如果不存在）
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
            # 复制文件
            shutil.copy(src_path, dst_path)
            print(f"File copied to {dst_path}")
    return


if __name__ == '__main__':
    EvtNumTrain = 1000
    file_path = "/Users/Sevati/PycharmProjects/untitled/PID/pid_data/MCTdata/hit_2.txt"
    df_train = load_data(file_path)###
    train_data = {}
    train_labels = {}
    all_efficiencies_1,all_efficiencies_2 = [],[]
    all_purities_1,all_purities_2 = [],[]
    successful_classes_1, total_classes_1 = 0, 0
    successful_classes_2, total_classes_2 = 0, 0
    for evtCount in range(EvtNumTrain):
        print(f'processing event======================: {evtCount}')
        hits = get_hits(df_train, evtCount)
        if hits.shape[0] < 10:
            continue


        train_data[evtCount] = hits[['finalX', 'finalY']].values
        train_labels[evtCount] = hits['trkid'].values

        eps = 0.1242#         0.1#mct           
        min_samples = 25#            1#mct            
        dbscan_clustering(hits,eps, min_samples)

        quantile = 0.5#           0.4683 #mct           
        n_samples = 100#                53#mct           
        meanshift_clustering(hits, quantile, n_samples)

        successful_classes_1, total_classes_1, unique_data = evalute_clusters(hits,evtCount, all_efficiencies_1, all_purities_1, successful_classes_1, total_classes_1,'dbscan_label')
        successful_classes_2, total_classes_2, unique_data = evalute_clusters(hits,evtCount, all_efficiencies_2, all_purities_2, successful_classes_2, total_classes_2,'meanshift_label')
        
        visualize_clusters(hits, unique_data)
        # pick_failure(hits)

    
    efficiency_1 = successful_classes_1/total_classes_1
    efficiency_2 = successful_classes_2/total_classes_2

    # 计算所有事例所有类的效率和纯度的均值
    mean_efficiency_db = np.mean(all_efficiencies_1)
    mean_purity_db = np.mean(all_purities_1)
    print(f"dbscan效率均值: {mean_efficiency_db:.2f}")
    print(f"dbscan纯度均值: {mean_purity_db:.2f}")
    print(f"dbscan成功率: {efficiency_1:.2f}")

    mean_efficiency_ms = np.mean(all_efficiencies_2)
    mean_purity_ms = np.mean(all_purities_2)
    print(f"meanshift效率均值: {mean_efficiency_ms:.2f}")
    print(f"meanshift纯度均值: {mean_purity_ms:.2f}")
    print(f"meanshift成功率: {efficiency_2:.2f}")


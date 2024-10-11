import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from math import sin,cos
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth,OPTICS
import matplotlib.cm as cm


def load_data(inputFile):
  # Load the data
  name=['evtid','trkid','layer','wire','x','y','rt','tdc']
  df=pd.read_table(inputFile,names=name,sep=',',header=None)
  return df


def get_hits(data,evtCount):
  df = data[data['evtid']==evtCount]
  raw = df.loc[(df['trkid'] > 0)]
  raw = raw.reset_index(drop=True)
  FinalX =[]
  FinalY =[]
  for i in list(raw.index):
     temp = raw['x'][i]*raw['x'][i] + raw['y'][i]*raw['y'][i]
     trans_x = 2*raw['x'][i] / temp
     trans_y = 2*raw['y'][i] / temp
     alpha = np.arctan2(trans_y,trans_x)
     FinalX.append( cos(alpha) )
     FinalY.append( sin(alpha) )
  raw['finalX'] = FinalX
  raw['finalY'] = FinalY
  return raw


def select_minpts(data_set):
    k=4
    k_dist = []
    for i in range(data_set.shape[0]):
        dist =  (((data_set[i] - data_set)**2).sum(axis=1)**0.5)
        dist.sort()
        k_dist.append(dist[k])
    k_dist = np.array(k_dist)
    k_dist.sort()
    delta = []
    for i in range(len(k_dist)-1):
       delta.append(k_dist[i+1]-k_dist[i])
    idx = delta.index(max(delta))
    return k_dist[idx]


def dbscan_clustering(data, eps=0.2, min_samples=4):
    # Extract x and y coordinates
    X = data[['finalX', 'finalY']].values

    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Add the labels back to the data to keep the original point order
    data['dbscan_label'] = labels

    # Number of clusters in labels, ignoring noise (-1 label)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return data, n_clusters_


def cluster(data, eps, min_pts):
    db = DBSCAN(eps=eps, min_samples=min_pts).fit(data[['finalX', 'finalY']])
    labels = db.labels_
    data['tag']= labels
    cluster_list = list(set(labels))
    if -1 in labels:
        cluster_list.remove(-1)
    return labels, cluster_list


def trans(data):
    FinalX =[]
    FinalY =[]
    for i in list(data.index):
        temp = data['x'][i]*data['x'][i] + data['y'][i]*data['y'][i]
        trans_x = 2*data['x'][i] / temp
        trans_y = 2*data['y'][i] / temp
        alpha = np.arctan2(trans_y,trans_x)
        FinalX.append( cos(alpha) )
        FinalY.append( sin(alpha) )
    data['finalX'] = FinalX
    data['finalY'] = FinalY
    return data


def run_DBSCAN(Hits, evtid):
    Hits_trans = trans(Hits)
    eps = select_minpts(Hits[['finalX', 'finalY']].values.reshape(Hits.shape[0], 2))
    try:
        Hits['tag'], cluster_list = cluster(Hits_trans, eps, 3)
    except:
        return 0, 0, True


def optimal_quantile(data, n_samples=100, quantiles=np.arange(0.1, 1.0, 0.1)):
    best_quantile = None
    best_num_clusters = 0
    
    # 计算每个点到原点的距离，作为带宽的候选数据
    distances = np.linalg.norm(data, axis=1)
    
    for q in quantiles:
        bandwidth = np.quantile(distances, q)
        mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        mean_shift.fit(data)
        
        num_clusters = len(np.unique(mean_shift.labels_))
        
        if num_clusters > best_num_clusters:
            best_num_clusters = num_clusters
            best_quantile = q
            
    return best_quantile


def meanshift_clustering(data, quantile=0.5, n_samples=100):
    # Extract x and y coordinates
    X = data[['finalX', 'finalY']].values
    # Estimate bandwidth
    bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)
    # Apply Mean Shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_

    # Add the labels back to the data to keep the original point order
    data['meanshift_label'] = labels

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print('Estimated number of clusters (Mean Shift): %d' % n_clusters_)
    return labels, n_clusters_


def reassign_labels(labels):
    label_mapping = {}  # 用来记录已经出现的标签及其新编号
    current_label = 0  # 当前可以分配的新编号
    new_labels = []  # 用来存放重新编号后的标签
    
    for label in labels:
        if label not in label_mapping:
            label_mapping[label] = current_label
            current_label += 1
        new_labels.append(label_mapping[label])
    
    return new_labels

from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
#计算效率和纯度
def calculate_efficiency_and_purity(gt_labels, dbscan_labels):
    unique_labels = np.unique(gt_labels)  # 包括所有标签，不过滤噪声
    results = {}
    successful_classes = 0

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

        # print(f"DBSCAN Class {label} has {correct_count} correct pairs out of {len(db_indices)} points. Efficiency: {efficiency:.2f}, Purity: {purity:.2f}")

        results[label] = {'efficiency': efficiency, 'purity': purity}

        # 判断寻类是否成功
        if efficiency > 0.8 and purity > 0.6:
            successful_classes += 1

    # 计算寻类成功率
    success_rate = successful_classes / len(unique_labels) if len(unique_labels) > 0 else 0
    # print(f"Class discovery success rate: {success_rate:.2f}")

    return results


# 计算成功率
def calculate_success_rate(gt_labels, method_labels, out_successful_classes, out_total_classes):
    # 计算效率和纯度
    results = calculate_efficiency_and_purity(gt_labels, method_labels)
    # print(f'MeanShift Clustering Results: {results}')
    successful_classes = 0
    total_classes = len(np.unique(gt_labels))  # 包括所有真实类（噪声也算）

    for label, metrics in results.items():
        efficiency = metrics['efficiency']
        purity = metrics['purity']

        if efficiency > 0.8 and purity > 0.6:
            successful_classes += 1

    success_rate = successful_classes / total_classes if total_classes > 0 else 0
    out_successful_classes += successful_classes
    out_total_classes += total_classes
    return success_rate, out_successful_classes, out_total_classes



def visualize_clusters(data, evtCount, successful_classes_1, total_classes_1,successful_classes_2, total_classes_2):
    print(f'processing event======================: {evtCount}')

    unique_trkids = data['trkid'].unique()
    trkid_colors = cm.rainbow(np.linspace(0, 1, len(unique_trkids)))
    gt_labels = data['trkid'].values

    dbscan_labels = data['dbscan_label'].values
    # dbscan_labels = data['tag'].values    
    meanshift_labels = data['meanshift_label'].values

    gt_labels = reassign_labels(gt_labels)
    dbscan_labels = reassign_labels(dbscan_labels)
    meanshift_labels = reassign_labels(meanshift_labels)
    
    tracking_efficiency1,successful_classes_1, total_classes_1 = calculate_success_rate(gt_labels, dbscan_labels,successful_classes_1, total_classes_1)
    tracking_efficiency2,successful_classes_2, total_classes_2 = calculate_success_rate(gt_labels, meanshift_labels,successful_classes_2, total_classes_2)

    # print(f'DBSCAN and MeanShift Tracking Efficiency: {tracking_efficiency1:.2f},     {tracking_efficiency2:.2f}')
    return successful_classes_1, total_classes_1, successful_classes_2, total_classes_2 


def main():
  df = load_data(inputFile)
  successful_classes_1, total_classes_1 = 0, 0
  successful_classes_2, total_classes_2 = 0, 0
  for evtCount in range(EvtNum):
    Hits = get_hits(df,evtCount) 
    if Hits.shape[0] < 10 :  
      continue
    
    # run_DBSCAN(Hits, evtCount)
    data_set = [Hits.finalX,Hits.finalY]
    data_set = np.array(data_set).T
    eps = 0.2#select_minpts(data_set)
    min_samples = 4
    data, dbscan_n_clusters_ = dbscan_clustering(Hits,eps, min_samples)

    quantile = 0.5#optimal_quantile(data_set)
    print(f'Optimal quantile: {quantile:.2f}')
    data, meanshift_n_clusters_ = meanshift_clustering(Hits, quantile=quantile, n_samples=100)
    
    successful_classes_1, total_classes_1, successful_classes_2, total_classes_2 = visualize_clusters(Hits,evtCount,successful_classes_1, total_classes_1,successful_classes_2, total_classes_2)
  print(f'All Tracking Success Rate: DBSCAN--{successful_classes_1}/{total_classes_1}={successful_classes_1/total_classes_1:.2f},     MeanShift--{successful_classes_2}/{total_classes_2}={successful_classes_2/total_classes_2:.2f}')
#   print(f'All DBSCAN and MeanShift : {successful_classes_1/total_classes_1:.2f},     {successful_classes_2/total_classes_2:.2f}')
      

    

if __name__ == '__main__':
  EvtNum = 10000
  inputFile = "/Users/Sevati/PycharmProjects/untitled/PID/pid_data/2Ddata/hit_5.txt"
  main()


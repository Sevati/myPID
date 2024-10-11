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


def dbscan_clustering(data, eps=0.05, min_samples=30):
    # Extract x and y coordinates
    X = data[['finalX', 'finalX']].values
    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print('Estimated number of clusters (DBSCAN): %d' % n_clusters_)  
    return labels, n_clusters_


def meanshift_clustering(data, quantile=0.5, n_samples=300):
    # Extract x and y coordinates
    X = data[['finalX', 'finalY']].values
    # Estimate bandwidth
    bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)
    # print(f'Estimated bandwidth: {bandwidth}')
    # Apply Mean Shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print('Estimated number of clusters (Mean Shift): %d' % n_clusters_)
    return labels, n_clusters_


def optics_clustering(data, min_samples=30):
    # Extract x and y coordinates
    X = data[['finalX', 'finalY']].values
    # Apply OPTICS
    optics = OPTICS(min_samples=min_samples).fit(X)
    labels = optics.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print('Estimated number of clusters (OPTICS): %d' % n_clusters_)
    return labels, n_clusters_


from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
def calculate_clustering_accuracy_via_hungarian(dbscan_labels, groundtruth_labels):
    # 将标签转换为numpy数组
    dbscan_labels = np.array(dbscan_labels)
    groundtruth_labels = np.array(groundtruth_labels)

    # 确保输入是有效的
    if len(dbscan_labels) != len(groundtruth_labels):
        raise ValueError("长度不匹配：dbscan_labels 和 groundtruth_labels 必须有相同长度")

    # 创建混淆矩阵
    conf_matrix = confusion_matrix(groundtruth_labels, dbscan_labels)

    # 使用匈牙利算法寻找最优标签对齐
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)

    # 计算经过优化后的ARI
    new_dbscan_labels = np.zeros_like(dbscan_labels)
    for i, j in zip(row_ind, col_ind):
        new_dbscan_labels[dbscan_labels == j] = i

    # 计算和返回调整兰德指数（Adjusted Rand Index）
    ari = adjusted_rand_score(groundtruth_labels, new_dbscan_labels)
    return ari


def visualize_clusters(data, dbscan_labels, dbscan_n_clusters_, meanshift_labels, meanshift_n_clusters_, evtCount):
    # Create a figure with 2x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(30, 16))  # Adjust figsize to make the figure smaller
    #12,8

    unique_trkids = data['trkid'].unique()
    trkid_colors = cm.rainbow(np.linspace(0, 1, len(unique_trkids)))
    # trkid_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_trkids)))#Matplotlib 提供了多种颜色映射，例如 viridis、plasma、inferno、magma、cividis 等。Spectral
    accuracy1 = calculate_clustering_accuracy_via_hungarian(dbscan_labels, data['trkid'])
    accuracy2 = calculate_clustering_accuracy_via_hungarian(meanshift_labels, data['trkid'])
    # accuracy = calculate_purity(dbscan_labels, unique_trkids)
    print(f'processing event: {evtCount}')
    print(f'DBSCAN and MeanShift Clustering Accuracy: {accuracy1:.2f},     {accuracy2:.2f}')
    

#显示结果子图2*3
'''
    # Subplot 1: Groundtruth labels (x, y)
    handles = []
    for k, col in zip(unique_trkids, trkid_colors):
        class_member_mask = (data['trkid'] == k)
        xy = data[class_member_mask]
        handle = axs[0, 0].scatter(xy['x'], xy['y'], s=30, color=col, label=f'Cluster {k} ({np.sum(data["trkid"] == k)})',edgecolor='k')
        handles.append(handle)
    axs[0, 0].set_title('Groundtruth Labels (x, y) - Event ' + str(evtCount))
    axs[0, 0].legend(handles=handles, loc='upper right')
    
    # Subplot 2: DBSCAN clustering results
    unique_dbscan_labels = set(dbscan_labels)
    dbscan_colors = cm.rainbow(np.linspace(0, 1, len(unique_dbscan_labels)))
    handles = []
    for k, col in zip(unique_dbscan_labels, dbscan_colors):
        class_member_mask = (dbscan_labels == k)
        xy = data[class_member_mask]
        handle = axs[0,1 ].scatter(xy['x'], xy['y'], s=30, color=col, label=f'Cluster {k} ({np.sum(dbscan_labels == k)})',edgecolor='k')
        handles.append(handle)
    axs[0,1].set_title('DBSCAN Clustering Results - Event ' + str(evtCount))
    axs[0,1].legend(handles=handles, loc='upper right')
    
    # Subplot 3: Mean Shift clustering results
    unique_meanshift_labels = set(meanshift_labels) 
    meanshift_colors = cm.rainbow(np.linspace(0, 1, len(unique_meanshift_labels)))
    handles = []
    for k, col in zip(unique_meanshift_labels, meanshift_colors):
        class_member_mask = (meanshift_labels == k)
        xy = data[class_member_mask]
        handle = axs[0,2].scatter(xy['x'], xy['y'], s=30, color=col, label=f'Cluster {k} ({np.sum(meanshift_labels == k)})',edgecolor='k')
        handles.append(handle)
    axs[0,2].set_title('Mean Shift Clustering Results - Event ' + str(evtCount))
    axs[0,2].legend(handles=handles, loc='upper right')
    
    # Subplot 4: Groundtruth labels (finalX, finalY)
    handles = []
    for k, col in zip(unique_trkids, trkid_colors):
        class_member_mask = (data['trkid'] == k)
        xy = data[class_member_mask]
        handle = axs[1,0].scatter(xy['finalX'], xy['finalY'], s=30, color=col, label=f'Cluster {k} ({np.sum(data["trkid"] == k)})',edgecolor='k')
        handles.append(handle)
    axs[1,0].set_title('Groundtruth Labels (finalX, finalY) - Event ' + str(evtCount))
    axs[1,0].legend(handles=handles, loc='upper right')
    
    # Subplot 5: DBSCAN clustering results (finalX, finalY)
    unique_dbscan_labels = set(dbscan_labels)
    dbscan_colors = cm.rainbow(np.linspace(0, 1, len(unique_dbscan_labels)))
    handles = []
    for k, col in zip(unique_dbscan_labels, dbscan_colors):
        class_member_mask = (dbscan_labels == k)
        xy = data[class_member_mask]
        handle = axs[1, 1].scatter(xy['finalX'], xy['finalY'], s=30, color=col, label=f'Cluster {k} ({np.sum(dbscan_labels == k)})',edgecolor='k')
        handles.append(handle)
    axs[1, 1].text(0.05, 0.05, f'Accuracy: {accuracy1:.2f}', transform=axs[1, 1].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    axs[1,1].set_title('DBSCAN Clustering Results (finalX, finalY) - Event ' + str(evtCount))
    axs[1,1].legend(handles=handles, loc='upper right')
    
    # Subplot 6: Mean Shift clustering results (finalX, finalY)
    unique_meanshift_labels = set(meanshift_labels)
    meanshift_colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_meanshift_labels)))
    handles = []
    for k, col in zip(unique_meanshift_labels, meanshift_colors):
        class_member_mask = (meanshift_labels == k)
        xy = data[class_member_mask]
        handle = axs[1, 2].scatter(xy['finalX'], xy['finalY'], s=30 , color=col, label=f'Cluster {k} ({np.sum(meanshift_labels == k)})',edgecolor='k')
        handles.append(handle)
    axs[1, 2].text(0.05, 0.05, f'Accuracy: {accuracy2:.2f}', transform=axs[1, 2].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    axs[1,2].set_title('Mean Shift Clustering Results (finalX, finalY) - Event ' + str(evtCount))
    axs[1,2].legend(handles=handles, loc='upper right')
    
    plt.tight_layout()  # Adjust layout to fit the figure better
    plt.savefig('/Users/Sevati/PycharmProjects/untitled/PID/out_plt/event' + str(evtCount) + '.jpg')

    # plt.show()
    # plt.close()
'''

def main():
  df = load_data(inputFile)
  for evtCount in range(EvtNum):
    Hits = get_hits(df,evtCount) 
    if Hits.shape[0] < 10 :  
      continue
    # next step is cluster based on "Hits", col:x,y is hits'position,same trkid means same track
    # finalX,finalY are the coordinate of the parameter space, and clustering can also be done in this space
    # print(Hits.finalX)
    # print(Hits.finalY)
    # print(len(Hits))
    dbscan_labels, dbscan_n_clusters_ = dbscan_clustering(Hits)
    meanshift_labels, meanshift_n_clusters_ = meanshift_clustering(Hits)
    optics_labels, optics_n_clusters_ = optics_clustering(Hits)
    visualize_clusters(Hits, dbscan_labels, dbscan_n_clusters_, meanshift_labels, meanshift_n_clusters_, evtCount)
    

if __name__ == '__main__':
  EvtNum = 100 
  inputFile = "/Users/Sevati/PycharmProjects/untitled/PID/pid_data/v2/hit_2.txt"
  main()


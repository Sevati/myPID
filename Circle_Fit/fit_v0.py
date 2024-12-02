import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
# from bayesian_optimization import BayesianOptimization
from scipy.optimize import minimize
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from matplotlib.patches import Arc
import math

import os
import sys
current_script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_script_path, '/Users/Sevati/PycharmProjects/untitled/PID/ML/test'))
from LoadData import load_data, get_hits, processing_data, reassign_labels
# try:
#     from LoadData import load_data
# except ImportError as e:
#     print("Error importing LoadData module: ", e)
#     sys.exit(1)

EvtNumTrain = 10
file_path = "/Users/Sevati/PycharmProjects/untitled/PID/pid_data/MCTdata/hit_2.txt"
df_train = load_data(file_path)
# 定义公差
radius_tolerance = 50  # 半径容许偏差
center_tolerance = 50  # 圆心容许距离
tolerance = {'center': 50, 'radius': 50}


def recursive_merge(circle_params, tolerance):
    ''' 递归合并相近的圆圈参数 '''
    merge_occurred = False
    n = len(circle_params)
    
    # 创建一个合并映射
    merge_map = list(range(n))
    
    for i in range(n):
        for j in range(i + 1, n):
            x0_i, y0_i, r_i = circle_params[i][:3]
            x0_j, y0_j, r_j = circle_params[j][:3]
            
            if (abs(x0_i - x0_j) < tolerance['center'] and 
                abs(y0_i - y0_j) < tolerance['center'] and 
                abs(r_i - r_j) < tolerance['radius']):
                
                merge_map[j] = merge_map[i]
                merge_occurred = True
    
    # 生成新的圆参数列表
    merged_circle_params = []
    for idx in set(merge_map):
        indices = [i for i, x in enumerate(merge_map) if x == idx]
        cluster_points = np.concatenate([circle_params[i][3] for i in indices], axis=0)
        x, y = cluster_points[:, 0], cluster_points[:, 1]
        x0, y0, r = initial_guess(x, y)
        merged_circle_params.append((x0, y0, r, cluster_points))

    if merge_occurred:
        return recursive_merge(merged_circle_params, tolerance)
    else:
        return merged_circle_params


def circle_residuals(params, x, y):
    x0, y0, r = params
    return np.sqrt((x - x0)**2 + (y - y0)**2) - r


def initial_guess(x, y):
    x_m, y_m = np.mean(x), np.mean(y)
    initial_guess = [x_m, y_m, np.sqrt((x - x_m)**2 + (y - y_m)**2).mean()]
    result = least_squares(circle_residuals, initial_guess, args=(x, y))
    x0, y0, r = result.x
    return x0, y0, r


def plot_circles(ax2, circle_params):
    ''' 在轴上绘制圆弧 '''
    for x0, y0, r, cluster_points in circle_params:
        angles = np.arctan2(cluster_points[:, 1] - y0, cluster_points[:, 0] - x0) * 180 / np.pi
        theta1, theta2 = np.min(angles), np.max(angles)
        arc = Arc([x0, y0], 2*r, 2*r, angle=0, theta1=theta1, theta2=theta2, color=col, linewidth=2)
        ax2.add_patch(arc)
    plt.scatter(coords[:,0], coords[:,1], color='blue', marker='.', label='Original Points')
        # 设置图示
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.set_title('Circle Fitting to the Given Points')
    ax2.legend()


if __name__ == '__main__':
    for evtCount in range(EvtNumTrain):
        hits = get_hits(df_train, evtCount)
        coords = hits[['x', 'y']].values
        para_coords = hits[['finalX', 'finalY']].values
        mct_labels = hits['trkid'].values
        bandwidth = estimate_bandwidth(para_coords, quantile=0.3, n_samples=200)
        # 应用MeanShift算法
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(para_coords)
        labels = ms.labels_
        # cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        print('Estimated number of clusters: %d' % n_clusters)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        #绘制子图1：聚类结果
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters), colors):
            my_members = labels == k
            # cluster_center = cluster_center[k]
            ax1.plot(coords[my_members, 0], coords[my_members, 1], col + '.')
            # ax1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            #          markeredgecolor='k', markersize=14)
        ax1.set_title('Clustering Results')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.set_aspect('equal', adjustable='datalim')
       
        # 初始化圆心和半径列表
        circle_params = []
        merged_labels = labels.copy()
        
        # 初步拟合每个簇的数据点为圆弧
        for k, col in zip(range(n_clusters), colors):
            cluster_points = coords[labels == k]
            x = cluster_points[:, 0]
            y = cluster_points[:, 1]
            x0, y0, r = initial_guess(x, y)
            print(f"Cluster {k}: Center=({x0}, {y0}), Radius={r}")
            circle_params.append((x0, y0, r, cluster_points))
        
        final_circles = recursive_merge(circle_params, tolerance)
        plot_circles(ax2, final_circles)
        ax2.set_aspect('equal', adjustable='datalim')
        # plt.show()
        plt.savefig('/Users/Sevati/PycharmProjects/untitled/PID/Axs_Results/axs_fit/event' + str(evtCount) + '.jpg')
        plt.close()






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


def circle_residuals(params, x, y):
    x0, y0, r = params
    return np.sqrt((x - x0)**2 + (y - y0)**2) - r


def initial_guess(x, y):
    x_m, y_m = np.mean(x), np.mean(y)
    initial_guess = [x_m, y_m, np.sqrt((x - x_m)**2 + (y - y_m)**2).mean()]
    result = least_squares(circle_residuals, initial_guess, args=(x, y))
    x0, y0, r = result.x
    return x0, y0, r


def plot_circle(x0, y0, r):
    theta = np.linspace(0, 2*np.pi, 100)
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    plt.plot(x, y, 'b-', label='Fitted Circle')


if __name__ == '__main__':
    for evtCount in range(EvtNumTrain):
        hits = get_hits(df_train, evtCount)
        coords = hits[['x', 'y']].values
        para_coords = hits[['finalX', 'finalY']].values
        
        mct_labels = hits['trkid'].values
        # 设定误差阈值
        threshold = 0.5
        
        bandwidth = estimate_bandwidth(para_coords, quantile=0.1, n_samples=500)
        # 应用MeanShift算法
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(para_coords)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        print('Estimated number of clusters: %d' % n_clusters)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        #绘制子图1：聚类结果
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            ax1.plot(coords[my_members, 0], coords[my_members, 1], col + '.')
            ax1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        ax1.set_title('Clustering Results')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.set_aspect('equal', adjustable='datalim')

        # 对每个簇的数据点拟合圆弧
        for k, col in zip(range(n_clusters), colors):
            cluster_points = coords[labels == k]
            x = cluster_points[:, 0]
            y = cluster_points[:, 1]
            x0, y0, r = initial_guess(x, y)
            print(f"Cluster {k}: Center=({x0}, {y0}), Radius={r}")
            angles = np.arctan2(y - y0, x - x0) * 180 / np.pi
            theta1, theta2 = np.min(angles), np.max(angles)
            arc = Arc([x0, y0], 2*r, 2*r, angle=0, theta1=theta1, theta2=theta2, color=col, linewidth=2)
            ax2.add_patch(arc)

        ax2.scatter(x, y, color='blue', label='Data Points')
        
        ax2.set_title('Fitted Circles to Each Cluster')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        
        plt.scatter(coords[:,0], coords[:,1], label='Original Points')

        
        # 设置图示
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        ax2.set_title('Circle Fitting to the Given Points')
        ax2.legend()

        # 使x和y轴等比例显示
        ax2.set_aspect('equal', adjustable='datalim')
        # 显示图形

        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig('/Users/Sevati/PycharmProjects/untitled/PID/Axs_Results/axs_fit/event' + str(evtCount) + '.jpg')
        plt.close()








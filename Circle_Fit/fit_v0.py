import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
# from bayesian_optimization import BayesianOptimization
from scipy.optimize import minimize
from sklearn.cluster import MeanShift, estimate_bandwidth


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

EvtNumTrain = 1
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


# # 定义圆弧方程的残差
# def residuals(params, coords):
#     x0, y0= params
#     x, y = coords.T
#     r = np.sqrt(x0**2 + y0**2)  # 半径等于圆心到原点的距离

#     return np.sqrt((x - x0)**2 + (y - y0)**2) - r

# # 拟合圆弧的函数
# def fit_circle(coords):
#     # 初始参数估计
#     x_m, y_m = np.mean(coords, axis=0)
#     initial_params = np.array([x_m, y_m])#, np.mean(np.sqrt((coords[:, 0] - x_m)**2 + (coords[:, 1] - y_m)**2))])
    
#     result = least_squares(residuals, initial_params, args=(coords,))
#     x0, y0 = result.x
#     return x0, y0, np.sqrt(result.x[0]**2 + result.x[1]**2)


# def calculate_residuals(circle_params, coords):
#     # 计算误差，每个点到圆的距离
#     x0, y0 = circle_params
#     x, y = coords.T
#     radial_distances = np.sqrt((x - x0)**2 + (y - y0)**2)
#     r = np.sqrt(x0**2 + y0**2)  # 圆心到原点的距离作为半径
#     return np.sum((radial_distances - r)**2)


# def fit_circle_to_origin(coords):
#     # 初始猜测：使用所有点的均值作为圆心的初始值
#     initial_guess = np.mean(coords, axis=0)

#     result = minimize(calculate_residuals, initial_guess, args=(coords,), method='BFGS')
    
#     x0, y0 = result.x
#     r = np.sqrt(x0**2 + y0**2)  # 计算拟合的半径
#     return x0, y0, r

# # 判断点是否在圆弧上（在误差范围内）
# def points_on_arc(coords, params, threshold):
#     x0, y0, r = params
#     x, y = coords.T
#     distances = np.sqrt((x - x0)**2 + (y - y0)**2)
#     return np.abs(distances - r) < threshold

def plot_circle(x0, y0, r):
    theta = np.linspace(0, 2*np.pi, 100)
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    plt.plot(x, y, 'b-', label='Fitted Circle')

if __name__ == '__main__':
    for evtCount in range(EvtNumTrain):
        hits = get_hits(df_train, evtCount)
        coords = hits[['tx', 'ty']].values

        mct_labels = hits['trkid'].values
        # 设定误差阈值
        threshold = 0.5
        
        bandwidth = estimate_bandwidth(coords, quantile=0.47, n_samples=53)
        # 应用MeanShift算法
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(coords)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        
        plt.title('Estimated number of clusters: %d' % n_clusters)
        plt.show()
        # 对每个簇的数据点拟合圆弧
        for k in range(n_clusters):
            cluster_points = coords[labels == k]
            x = cluster_points[:, 0]
            y = cluster_points[:, 1]
            x0, y0, r = initial_guess(x, y)
            print(f"Cluster {k}: Center=({x0}, {y0}), Radius={r}")

        # x = coords[:, 0]
        # y = coords[:, 1]

        # x0, y0, r = initial_guess(x, y)
        # # x0, y0 = params
        # print(f"Fitted circle parameters: Center=({x0}, {y0}), Radius={r}")

        # 迭代拟合圆弧并绘制结果
        # remaining_coords = coords.copy()
        # iteration = 1
        

        # while len(remaining_coords) > 0:
        #     params = fit_circle(remaining_coords)
        #     on_arc = points_on_arc(remaining_coords, params, threshold)
    
        #     if np.any(on_arc):
        #         plt.plot(
        #             [remaining_coords[on_arc, 0].min(), remaining_coords[on_arc, 0].max()],
        #             [remaining_coords[on_arc, 1].min(), remaining_coords[on_arc, 1].max()],
        #             label=f'Fit Arc {iteration}'
        #         )
        #         remaining_coords = remaining_coords[~on_arc]  # 删除已经拟合上的点
        #     else:
        #         break
        #     iteration += 1
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='blue', label='Data Points')

        plt.figure()
        plt.scatter(coords[:,0], coords[:,1], label='Original Points')

        # 绘制拟合的圆
        circle = plt.Circle((x0, y0), r, color='red', fill=False, linewidth=2, label='Fitted Circle')
        ax.add_artist(circle)
        # 设置图示
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Circle Fitting to the Given Points')
        ax.legend()

        # 使x和y轴等比例显示
        ax.set_aspect('equal', adjustable='datalim')
        plt.grid(True)

        # 显示图形
        plt.show()








import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

points = np.array([
    [-6.04194, -6.3774], [-6.9269, -7.2379], [-7.85083, -8.13188], [-8.65849, -8.90877],
    [-9.50426, -9.71848], [-10.3689, -10.5419], [-11.1681, -11.2991], [-13.9372, -13.8944],
    [-15.1335, -15.0029], [-16.35, -16.121], [-17.4992, -17.17], [-18.7738, -18.3245],
    [-19.9511, -19.3821], [-21.1733, -20.4723], [-22.3917, -21.551], [-23.607, -22.6194],
    [-24.7694, -23.6604], [-26.0291, -24.7259], [-27.2007, -25.734], [-28.9129, -27.1949],
    [-30.1075, -28.2054], [-31.2981, -29.2055], [-32.5775, -30.2723], [-33.8314, -31.3103],
    [-35.1009, -32.3533], [-36.3356, -33.3601], [-37.6816, -34.4495], [-40.2072, -36.4709]
])

def circle_residuals(params, x, y):
    x0, y0, r = params
    return np.sqrt((x - x0)**2 + (y - y0)**2) - r

x = points[:, 0]
y = points[:, 1]
x_m = np.mean(x)
y_m = np.mean(y)

# 计算最佳带宽
bandwidth = estimate_bandwidth(points, quantile=0.47, n_samples=53)

# 应用MeanShift算法
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(points)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters = len(labels_unique)
# 绘制结果
plt.figure()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

for k, col in zip(range(n_clusters), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(points[my_members, 0], points[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters)
plt.show()

# 对每个簇的数据点拟合圆弧
for k in range(n_clusters):
    cluster_points = points[labels == k]
    x = cluster_points[:, 0]
    y = cluster_points[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)
    initial_guess = [x_m, y_m, np.sqrt((x - x_m)**2 + (y - y_m)**2).mean()]
    result = least_squares(circle_residuals, initial_guess, args=(x, y))
    x0, y0, r = result.x
    print(f"Cluster {k}: Center=({x0}, {y0}), Radius={r}")


# 生成绘图
fig, ax = plt.subplots()

# 绘制散点
ax.scatter(x, y, color='blue', label='Data Points')

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


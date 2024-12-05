###########先用初始meanshift聚成大类，然后再用fit_arc拟合圆弧，最后计算置信度。
# 如果置信度大于阈值，则保留，否则降低quantile，重复聚类，直到所有圆弧的置信度都高于阈值。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.optimize import leastsq
from matplotlib.patches import Arc
from sklearn.preprocessing import LabelEncoder
import matplotlib
from scipy.spatial.distance import cdist

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
EvtNumTrain = 102
file_path = "/Users/Sevati/PycharmProjects/untitled/PID/pid_data/MCTdata/hit_2.txt"
df_train = load_data(file_path)
# 定义公差
radius_tolerance = 50  # 半径容许偏差
center_tolerance = 50  # 圆心容许距离
tolerance = {'center': 50, 'radius': 50}
max_retries = 5  # 最大重试次数

# Step 1: MeanShift聚类
def mean_shift_clustering(para_coords, quantile=0.3, n_samples=100):
    bandwidth = estimate_bandwidth(para_coords, quantile=quantile, n_samples=n_samples)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(para_coords)
    labels = ms.labels_
    return labels

def dbscan_clustering(para_coords, eps=0.124, min_samples=25):
    db = DBSCAN(eps=eps, min_samples=int(min_samples))
    db.fit(para_coords)
    labels = db.labels_
    return labels


# Step 2: 拟合圆弧
def fit_arc(points):
    # 如果点数少于5个，则跳过拟合
    if len(points) < 5:
        print("Not enough points for fitting (less than 5). Skipping this cluster.")
        return None  # 返回 None，表示无法拟合
    # 圆的拟合：最小二乘法拟合圆
    def residuals(params, x, y):
        xc, yc, r = params
        return (np.sqrt((x - xc)**2 + (y - yc)**2) - r)

    x = points[:, 0]
    y = points[:, 1]
    # 初始猜测：假设圆心在点的均值附近，半径为这些点到均值的平均距离
    initial_guess = [np.mean(x), np.mean(y), np.mean(np.sqrt(x**2 + y**2))]     #r = np.mean(np.sqrt(x**2 + y**2))
    params, _ = leastsq(residuals, initial_guess, args=(x, y))
    # print(params)
    xc, yc, r = params

    return xc, yc, r

# Step 3: 计算置信度
def compute_confidence(points, xc, yc, r):
    distances = np.abs(np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2) - r)
    avg_distance = np.mean(distances)
    confidence = 1 / (1 + avg_distance)  # 置信度与平均距离成反比
    return confidence, avg_distance

# Step 4: 调整 MeanShift 直到所有圆弧的置信度都高于阈值
# def process_clusters(hits, threshold, eps=0.124, min_samples=25):
def process_clusters(hits, threshold, initial_quantile=0.5, n_samples=100):
    points = hits[['x', 'y']].values
    para_coords = hits[['finalX', 'finalY']].values
    quantile = initial_quantile
    # min_samples = 25
    previous_confidences = None
    retries = 0
    while True:   #调参
        # 使用 MeanShift 聚类
        # labels = dbscan_clustering(para_coords, eps, min_samples)
        labels = mean_shift_clustering(para_coords, quantile=quantile, n_samples=n_samples)
        all_confidences = []
        track_data = {}
        
        # 对每个聚类，拟合圆弧并计算置信度
        for cluster_label in np.unique(labels):
            cluster_points = points[labels == cluster_label]
            if cluster_points.shape[0] < 10:
                continue
            xc, yc, r = fit_arc(cluster_points)
            confidence, _ = compute_confidence(cluster_points, xc, yc, r)
            all_confidences.append((cluster_label, xc, yc, r, confidence))
            track_data[cluster_label] = cluster_points  # 保存聚类的点

        # Step 1: 合并径迹（递归合并逻辑）
        if len(all_confidences) > 1:   
            merged_confidences = all_confidences.copy()
            merged_tracks = set()  # 用于存储已经合并过的径迹
            updated_labels = labels.copy()  # 创建一个新的标签数组来更新标签

            # 循环直到没有可合并的情况
            def merge_clusters_recursive(i):
                if i >= len(merged_confidences):
                    return

                label, xc, yc, r, confidence = merged_confidences[i]
                # 尝试合并当前标签与后续标签
                for j in range(i + 1, len(merged_confidences)):
                    other_label, other_xc, other_yc, other_r, other_confidence = merged_confidences[j]

                    if (label, other_label) not in merged_tracks:
                        merged_result = merge_tracks(track_data[label], track_data[other_label])

                        if merged_result is not None:
                            new_xc, new_yc, new_r = merged_result
                            merged_confidence, _ = compute_confidence(
                                np.vstack((points[labels == label], points[labels == other_label])),
                                new_xc, new_yc, new_r
                            )

                            # 更新合并后的结果
                            merged_confidences[:] = [conf for conf in merged_confidences if conf[0] != other_label and conf[0] != label]
                            merged_confidences.append((label, new_xc, new_yc, new_r, merged_confidence))

                            # 更新标签
                            updated_labels[updated_labels == other_label] = label  # 将第二个径迹的标签更新为第一个径迹的标签
                            merged_tracks.add((label, other_label))  # 标记这两条径迹已经合并

                            # 递归地继续合并合并后的径迹
                            merge_clusters_recursive(i)
                            return  # 如果合并了，就停止当前循环

                # 如果没有合并，则跳到下一个标签
                merge_clusters_recursive(i + 1)

            # 从第一个标签开始递归合并
            merge_clusters_recursive(0)

            labels = updated_labels.copy()
            all_confidences = merged_confidences
        #****************************************
        
        # Step 2: 检查是否所有圆弧的置信度都大于阈值
        all_above_threshold = all(confidence >= threshold for _, _, _, _, confidence in all_confidences)

        # 如果所有圆弧的置信度都高于阈值，退出
        if all_above_threshold:
            break

        # 检查置信度是否有提高
        current_confidences = [confidence for _, _, _, _, confidence in all_confidences]
        print(f"Current confidences: {current_confidences}")
        if previous_confidences is not None and current_confidences == previous_confidences:
            # 如果修改后置信度没有提高，恢复参数并终止
            print("No improvement in confidence after changing parameters. Restoring previous settings and terminating.")
            break
        else:
            # 更新 previous_confidences 并增加重试次数
            previous_confidences = current_confidences
            retries += 1

        # 如果已达到最大重试次数，则终止
        if retries >= max_retries:
            print("Maximum retries reached. Terminating.")
            break

        # 如果有圆弧的置信度低于阈值，则降低 quantile 重复聚类
        quantile -= 0.1
        quantile = max(0.1, quantile)  # 确保 quantile 不小于0.1
        # min_samples -= 1
        # min_samples = max(2, min_samples)  # 确保 min_samples 不小于2

    return labels, all_confidences


# Step 5: 判断两条径迹是否来自同一个圆，并合并
def merge_tracks(track1, track2, threshold_radius=10, threshold_center=10, threshold_distance=20):
    """
    判断两条径迹是否可能来自同一个圆, 如果是，则合并它们并重新拟合圆弧. 
    在判断两条径迹是否可能来自同一个圆时,考虑以下因素: 1.圆心和半径的偏差 2.径迹之间的整体距离 3.合并后平均置信度是否提高 
    参数：
    track1, track2: 两条径迹的数据点 (numpy 数组)
    threshold_radius: 圆的半径差异阈值
    threshold_center: 圆心的最大距离阈值
    threshold_distance: 两条径迹整体的点间距离阈值

    返回：
    合并后的拟合圆心和半径 (xc, yc, r) 或 None
    """

    # 拟合两条径迹的圆弧
    xc1, yc1, r1 = fit_arc(track1)
    xc2, yc2, r2 = fit_arc(track2)

    if xc1 is None or xc2 is None:
        return None  # 如果其中一条径迹无法拟合圆弧，则返回 None

    # 1. 判断圆心和半径的偏差是否足够小
    distance_between_centers = np.sqrt((xc1 - xc2)**2 + (yc1 - yc2)**2)
    radius_diff = np.abs(r1 - r2)

    if distance_between_centers < threshold_center and radius_diff < threshold_radius:
        print(f"Track1 and Track2 may come from the same circle based on center and radius.")
        merged_points = np.vstack((track1, track2))
        return fit_arc(merged_points)

    # 2. 判断两条径迹的点之间整体距离是否较小
    overall_distance = np.mean(cdist(track1, track2))  # 计算所有点的平均距离
    if overall_distance < threshold_distance:
        print(f"Track1 and Track2 may come from the same circle based on overall distance.")
        merged_points = np.vstack((track1, track2))
        return fit_arc(merged_points)
    
    # 3. 判断合并后平均置信度是否提高
    confidence1, _ = compute_confidence(track1, xc1, yc1, r1)
    confidence2, _ = compute_confidence(track2, xc2, yc2, r2)
    #合并后径迹的置信度
    merged_points = np.vstack((track1, track2))
    merged_xc, merged_yc, merged_r = fit_arc(merged_points)
    merged_confidence, _ = compute_confidence(merged_points, merged_xc, merged_yc, merged_r)
    # 如果合并后的置信度提高，认为可以合并
    average_initial_confidence = (confidence1 + confidence2) / 2
    if merged_confidence > average_initial_confidence:
        print(f"Track1 and Track2 merged due to improved confidence.")
        return merged_xc, merged_yc, merged_r
    
    # 如果所有条件都不满足，则返回 None
    return None


# 可视化
def visualize_clusters(evtCount, points, labels, confidences):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 设定一个颜色映射（Color Map）
    le = LabelEncoder()
    unique_labels = le.fit_transform(labels)  # 对标签进行编码
    
    # 创建一个包含len(np.unique(labels))个颜色的tab10色图
    cmap = matplotlib.colormaps['tab10'].colors
    cmap = matplotlib.colors.ListedColormap(cmap[:len(np.unique(labels))])  # 根据类别数量切片


    # 绘制半透明的散点，边框不透明
    ax.scatter(points[:, 0], points[:, 1], c=unique_labels, cmap=cmap, label='Data Points', 
               alpha=0.3, edgecolor='black', linewidth=0.5)
    
    for cluster_label, xc, yc, r, confidence in confidences:
        if confidence > 0:  # 只画置信度高的圆弧
            # 获取属于该聚类的点
            cluster_points = points[labels == cluster_label]
            
            # 计算每个点的角度
            angles = np.arctan2(cluster_points[:, 1] - yc, cluster_points[:, 0] - xc) * 180 / np.pi
            theta1, theta2 = np.min(angles), np.max(angles)

            # 获取对应聚类的颜色
            cluster_color = cmap(cluster_label % len(np.unique(labels)))

            # 计算圆弧的x, y坐标
            theta = np.linspace(np.radians(theta1), np.radians(theta2), 100)
            x = xc + r * np.cos(theta)
            y = yc + r * np.sin(theta)

            # 使用 plt.plot 显示圆弧，并添加置信度到图例
            ax.plot(x, y, label=f'Cluster {cluster_label} (Confidence: {confidence:.2f})', color=cluster_color, linewidth=2)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    # plt.show()
    plt.savefig('/Users/Sevati/PycharmProjects/untitled/PID/Axs_Results/axs_fit_ms/event' + str(evtCount) + '.jpg')
    plt.close()


# 示例使用
if __name__ == "__main__":
    for evtCount in range (7,8):#(EvtNumTrain):
        print("-----------Processing event-----------:", evtCount)
        hits = get_hits(df_train, evtCount)
        coords = hits[['x', 'y']].values
        # para_coords = hits[['finalX', 'finalY']].values
        # 处理聚类并拟合圆弧
        labels, confidences = process_clusters(hits, threshold=0.15)   #置信度阈值
        
        # 可视化结果
        visualize_clusters(evtCount, coords, labels, confidences)

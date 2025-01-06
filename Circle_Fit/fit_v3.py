###########在v2的基础上增加了ransac,当某类点数大于60或某两类距离过近，则用ransac重新拟合。

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

cluster_method = 'meanshift' #'dbscan' or 'meanshift' or 'mct'
data_type = 'hit' # 'hit' or 'truth'

EvtNumTrain = 1000
# file_path = '/Users/Sevati/PycharmProjects/untitled/PID/pid_data/getFildEvent/GenfitOut_2.txt'
file_path = '/Users/Sevati/PycharmProjects/untitled/PID/pid_data/MCTdata/hit_2.txt'
df_train = load_data(file_path)
# 定义公差
# radius_tolerance = 50  # 半径容许偏差
# center_tolerance = 50  # 圆心容许距离
tolerance = {'center': 10, 'radius': 10, 'dist': 20}
max_retries = 5  # 最大重试次数
threshold = 0.2 if data_type == 'hit' else 0.6   #置信度阈值

# Step 1: MeanShift聚类
def mean_shift_clustering(hits, quantile=0.3, n_samples=100):
    para_coords = hits[['finalX', 'finalY']].values if data_type == 'hit' else hits[['finalTX', 'finalTY']].values
    bandwidth = estimate_bandwidth(para_coords, quantile=quantile, n_samples=n_samples)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(para_coords)#, sample_weight=hits['weight'])
    labels = ms.labels_
    hits['label'] = reassign_labels(labels)
    #*******************merge cluster*************************
    # Hits = Hits[Hits['label'] != False].reset_index(drop=True)#no noise
    merged = True
    while merged:
        hits, merged = merge_labels(hits)
    #**********************************************************
    
    return hits

def dbscan_clustering(hits, eps=0.124, min_samples=25):
    para_coords = hits[['finalX', 'finalY']].values if data_type == 'hit' else hits[['finalTX', 'finalTY']].values    
    db = DBSCAN(eps=eps, min_samples=int(min_samples))
    db.fit(para_coords)
    labels = db.labels_
    hits['label'] = reassign_labels(labels)
    #*******************merge cluster*************************
    # Hits = Hits[Hits['label'] != False].reset_index(drop=True)#no noise
    merged = True
    while merged:
        hits, merged = merge_labels(hits)
    #**********************************************************
    
    return hits

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

# 判断是否需要重新聚类的函数
def need_recluster(hits,threshold=25, max_class_size=60):
    labels = hits['label']
    # 提取各类的点数量
    class_sizes = {label: np.sum(labels == label) for label in np.unique(labels)}
    
    # 检查条件：某类点个数超过 max_class_size 或者某两类的点平均距离小于 threshold
    for label1 in class_sizes:
        # 判断当前类的点是否超过 max_class_size
        if class_sizes[label1] > max_class_size:
            return True, hits
        
    # 如果标签的数量小于 2，则不需要判断两类点的平均距离
    if len(class_sizes) < 2:
        return False, hits
    
    # 检查两类点的平均距离
    for label1 in class_sizes:
        for label2 in class_sizes:
            # if label1 != label2:
            if label1 != label2 and class_sizes[label1] > 10 and class_sizes[label2] > 10:
                # 获取label1和label2的点
                points1 = hits[labels == label1]
                points2 = hits[labels == label2]
                # 提取坐标
                points1_xy = points1[['x', 'y']].values
                points2_xy = points2[['x', 'y']].values

                distances = cdist(points1_xy[:, :2], points2_xy[:, :2], 'euclidean')
                
                # 计算两个类的平均距离
                avg_distance = np.mean(distances)
                
                # 如果平均距离小于阈值，则需要重新聚类
                if avg_distance < threshold:
                    hits.loc[labels == label2, 'label'] = label1
                    return True, hits
    return False, hits

# 执行重新聚类的函数
def re_cluster_with_ransac(hits):
    hits['R']=''  # R
    hits['Phi']=''  # Phi
    # 计算极坐标 R 和 Phi
    hits['R'] = np.sqrt(hits['x']**2 + hits['y']**2)
    hits['Phi'] = np.arctan2(hits['y'], hits['x'])
    # labels = hits['label']
    # # 提取符合条件的点
    # subhits = hits[np.isin(labels, np.unique(labels))]
    subhits = hits.copy()

    label=1
    hits['label'] = False
    while(subhits.shape[0] > 15 ):
        subhits = RANSAC(subhits)
        subhits.loc[subhits['label'] == True, 'label'] = label
        hits.loc[hits['label'] == False, 'label'] = subhits['label']
     
        subhits = hits[hits['label'] ==False]
        label += 1
    hits.loc[hits['label'] == False, 'label'] = label #have noise
    #remove noise class,layer >3
    hits = hits.groupby('label').filter(lambda x: x['layer'].nunique() > 3 )
    hits = hits.reset_index(drop=True)

    #*******************merge cluster*************************
    # Hits = Hits[Hits['label'] != False].reset_index(drop=True)#no noise
    merged = True
    while merged:
        hits, merged = merge_labels(hits)
    #**********************************************************
    hits['label'] = reassign_labels(hits['label'])

    return hits

#用ransac聚类
from sklearn.linear_model import RANSACRegressor
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def RANSAC(Hits):
  R = Hits['R'].values.reshape(-1,1)
  Phi = Hits['Phi'].values.reshape(-1,1)
  ransac = linear_model.RANSACRegressor(random_state=0)
  ransac.fit(R,Phi)
  inlier_mask = ransac.inlier_mask_
  Hits['label']=inlier_mask
  return Hits

def merge_labels(df):
  label_layers = df.groupby('label')['layer'].apply(set).reset_index()
  label_layers['length'] = label_layers['layer'].apply(len)
  label_layers = label_layers[label_layers['length'] >= 5]
  tag_list = label_layers.sort_values('length')['label'].tolist()
  group_combinations = []
  loop = 1
  for i in tag_list:  # i,j is tag
    candidate = []
    for j in tag_list[loop:]:
          common_layers = len(label_layers.loc[label_layers['label'] == i, 'layer'].values[0].intersection
                              (label_layers.loc[label_layers['label'] == j, 'layer'].values[0]))
          if common_layers < 5:  # same layers must <3 when merge
              candidate.append(j)
    if len(candidate) ==0:
       loop+=1
       continue
    elif len(candidate) ==1:
       df.loc[df['label'] == candidate[0], 'label'] = i
       #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$1')
       return df, True
    else:
       if 'Phi' not in df.columns:
         df['Phi']=''
         for m in range(df.shape[0]):
           x=df.loc[m,'x']
           y=df.loc[m,'y']
           df.loc[m,'Phi'] = np.arctan2(y,x)
       diff0 = 10
       phi_avg1 = df[df['label'] == i]['Phi'].mean()
       for k in candidate:
          phi_avg2 = df[df['label'] == k]['Phi'].mean()
          diff = abs(phi_avg1 - phi_avg2)
          if diff < diff0:
             diff0 = diff
             win = k
       df.loc[df['label'] == win, 'label'] = i
       #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$2')
       return df, True
  return df, False

def run_RANSAC(Hits):
  Hits['R']=''  # R
  Hits['Phi']=''  # Phi
  for i in range(Hits.shape[0]):
     x=Hits.loc[i,'x']
     y=Hits.loc[i,'y']
     Hits['Phi'] = np.arctan2(Hits['y'], Hits['x'])
     Hits['R'] = np.sqrt(Hits['x']**2 + Hits['y']**2)
  label=1
  subHits = Hits.copy()
  Hits['label'] = False
  while(subHits.shape[0] > 15):
     subHits = RANSAC(subHits)
     subHits.loc[subHits['label'] == True] = label
     Hits.loc[Hits['label'] == False, 'label'] = subHits['label']
     subHits = Hits[Hits['label'] ==False]
     label += 1
  Hits.loc[Hits['label'] == False, 'label'] = label #have noise
  #remove noise class,layer >3
#   Hits = Hits.groupby('label').filter(lambda x: x['layer'].nunique() > 3 )
#   Hits = Hits.reset_index(drop=True)
  return Hits,Hits['label'].values


# Step 4: 调整 MeanShift 直到所有圆弧的置信度都高于阈值
# def process_clusters(hits, eps=0.124, min_samples=25):
def process_clusters(hits):
    ## 0. 筛选出直丝层的数据
    # vertical_layers = list(range(8, 20)) + list(range(36, 43))
    # vertical_hits = hits[hits['layer'].isin(vertical_layers)]

    # para_coords = vertical_hits[['finalX', 'finalY']].values if data_type == 'hit' else hits[['finalTX', 'finalTY']].values

    previous_confidences = 0
    retries = 0
    if cluster_method == 'dbscan':
        eps = 0.124
        min_samples = 25
    elif cluster_method == 'meanshift':
        quantile=0.45
        n_samples=84


    while True:   #调参
        # Step 1: 聚类
        hits = mean_shift_clustering(hits, quantile=quantile, n_samples=n_samples) if cluster_method =='meanshift' else dbscan_clustering(hits, eps, min_samples) if cluster_method == 'dbscan' else hits
        hits['label'] = hits['trkid'] if cluster_method=='mct' else hits['label']

        all_confidences = []
        track_data = {}

        # Step 2: 检查是否所有径迹的点都小于60，若不是，则用ransac重新聚类
        need, hits = need_recluster(hits)
        if need: 
            subHits = hits.copy()
            hits = re_cluster_with_ransac(subHits)


        points = hits[['x', 'y']].values if data_type == 'hit' else hits[['tx', 'ty']].values
        # Step 3: 拟合圆弧并计算置信度
        for cluster_label in np.unique(hits['label']):
            # vertical_points = vertical_hits[['x', 'y']].values if data_type == 'hit' else vertical_hits[['tx', 'ty']].values
            # cluster_points = vertical_points[vertical_hits['label'] == cluster_label]
            cluster_points = points[hits['label'] == cluster_label]
            if cluster_points.shape[0] < 5:
                continue
            xc, yc, r = fit_arc(cluster_points)
            confidence, _ = compute_confidence(cluster_points, xc, yc, r)
            all_confidences.append((cluster_label, xc, yc, r, confidence))
            track_data[cluster_label] = cluster_points  # 保存聚类的点

        # Step 4: 合并径迹（递归合并逻辑）
        if len(all_confidences) > 1:   
            merged_confidences = all_confidences.copy()
            merged_tracks = set()  # 用于存储已经合并过的径迹
            updated_labels = hits['label'].copy()  # 创建一个新的标签数组来更新标签

            # 循环直到没有可合并的情况
            def merge_clusters_recursive(i):
                if i >= len(merged_confidences):
                    return

                label, xc, yc, r, confidence = merged_confidences[i]
                # 尝试合并当前标签与后续标签
                for j in range(i + 1, len(merged_confidences)):
                    other_label, other_xc, other_yc, other_r, other_confidence = merged_confidences[j]

                    if (label, other_label) not in merged_tracks:
                        merged_result = merge_tracks(track_data[label], track_data[other_label], tolerance) # 合并两条径迹

                        if merged_result is not None:
                            new_xc, new_yc, new_r = merged_result
                            merged_confidence, _ = compute_confidence(
                                np.vstack((points[hits['label'] == label], points[hits['label'] == other_label])),
                                new_xc, new_yc, new_r
                            )

                            # 更新合并后的结果
                            merged_confidences[:] = [conf for conf in merged_confidences if conf[0] != other_label and conf[0] != label]
                            merged_confidences.append((label, new_xc, new_yc, new_r, merged_confidence))

                            # 更新标签
                            updated_labels[updated_labels == other_label] = label  # 将第二个径迹的标签更新为第一个径迹的标签
                            merged_tracks.add((label, other_label))  # 标记这两条径迹已经合并

                            # 递归地继续合并 合并后的径迹
                            merge_clusters_recursive(i)#？？？？？？？？？？？
                            return  # 如果合并了，就停止当前循环

                # 如果没有合并，则跳到下一个标签
                merge_clusters_recursive(i + 1)

            # 从第一个标签开始递归合并
            merge_clusters_recursive(0)

            labels = updated_labels.copy()
            hits['label'] = labels#reassign_labels(labels)
            all_confidences = merged_confidences
        

        # Step 5: 检查是否所有圆弧的置信度都大于阈值
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

        # 如果有圆弧的置信度低于阈值，则减小参数使得类别更细并重复聚类
        if cluster_method == 'dbscan':
            min_samples -= 1
            min_samples = max(2, min_samples)  # 确保 min_samples 不小于2
        elif cluster_method == 'meanshift':
            quantile -= 0.1
            quantile = max(0.1, quantile)  # 确保 quantile 不小于0.1
    
    return hits, all_confidences


# Step 5: 判断两条径迹是否来自同一个圆，并合并
def merge_tracks(track1, track2, tolerance):#threshold_radius=10, threshold_center=10, threshold_distance=20):
    """
    判断两条径迹是否可能来自同一个圆, 如果是，则合并它们并重新拟合圆弧. 
    在判断两条径迹是否可能来自同一个圆时,考虑以下因素: 1.圆心和半径的偏差 2.径迹之间的整体距离 3.合并后置信度是否提高 
    参数：
    track1, track2: 两条径迹的数据点 (numpy 数组)
    threshold_radius: 圆的半径差异阈值
    threshold_center: 圆心的最大距离阈值
    threshold_distance: 两条径迹整体的点间距离阈值

    返回：
    合并后的拟合圆心和半径 (xc, yc, r) 或 None
    """
    threshold_radius, threshold_center, threshold_distance = tolerance['radius'], tolerance['center'], tolerance['dist']

    # 拟合两条径迹的圆弧
    xc1, yc1, r1 = fit_arc(track1)
    xc2, yc2, r2 = fit_arc(track2)

    if xc1 is None or xc2 is None:
        return None  # 如果其中一条径迹无法拟合圆弧，则返回 None

    # 1. 判断圆心和半径的偏差是否足够小
    distance_between_centers = np.sqrt((xc1 - xc2)**2 + (yc1 - yc2)**2)
    radius_diff = np.abs(r1 - r2)

    if distance_between_centers < threshold_center and radius_diff < threshold_radius:
        print(f"Track1 and Track2 may come from the same circle based on center and radius.-----1")
        merged_points = np.vstack((track1, track2))
        return fit_arc(merged_points)
    
    confidence1, _ = compute_confidence(track1, xc1, yc1, r1)
    confidence2, _ = compute_confidence(track2, xc2, yc2, r2)

    # # 2. 判断两条径迹的点之间整体距离是否较小
    # overall_distance = np.mean(cdist(track1, track2))  # 计算所有点的平均距离
    # if overall_distance < threshold_distance:
    #     print(f"Track1 and Track2 may come from the same circle based on overall distance.-----2")
    #     merged_points = np.vstack((track1, track2))
    #     return fit_arc(merged_points)
    
    # # 3. 判断合并后平均置信度是否提高
    # #合并后径迹的置信度
    # merged_points = np.vstack((track1, track2))
    # merged_xc, merged_yc, merged_r = fit_arc(merged_points)
    # merged_confidence, _ = compute_confidence(merged_points, merged_xc, merged_yc, merged_r)
    # # 如果合并后的置信度提高，认为可以合并
    # average_initial_confidence = (confidence1 + confidence2) / 2
    # if merged_confidence > average_initial_confidence:
    #     print(f"Merged due to improved confidence.-----3")
    #     return merged_xc, merged_yc, merged_r

    # #4. 判断较低置信度的径迹是否可以用较高置信度的径迹的圆心和半径来计算
    # if confidence1 >= confidence2:
    #     # 使用 track1 的圆心和半径计算 track2 的置信度
    #     track2_confidence, _ = compute_confidence(track2, xc1, yc1, r1)
    #     if track2_confidence > threshold:
    #         merged_points = np.vstack((track1, track2))
    #         print(f"Merged due to improved confidence.-----4")
    #         return fit_arc(merged_points)
    # else:
    #     # 使用 track2 的圆心和半径计算 track1 的置信度
    #     track1_confidence, _ = compute_confidence(track1, xc2, yc2, r2)
    #     if track1_confidence > threshold:
    #         merged_points = np.vstack((track1, track2))
    #         print(f"Merged due to improved confidence.-----4")
    #         return fit_arc(merged_points)
        

    # 如果所有条件都不满足，则返回 None
    return None


# 可视化
def visualize_clusters(evtCount, hits, confidences):
    points = hits[['x', 'y']].values if data_type == 'hit' else hits[['tx', 'ty']].values
    labels = hits['label']

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
            cluster_points = points[hits['label'] == cluster_label]
            # 计算每个点的角度
            angles = np.arctan2(cluster_points[:, 1] - yc, cluster_points[:, 0] - xc) * 180 / np.pi
        
            # 将负角度转换为正值
            angles_positive = np.where(angles < 0, angles + 360, angles)
            sorted_indices = np.argsort(angles_positive)
            angles_sorted = np.sort(angles_positive)
            # diffs = np.diff(angles_sorted)
            diffs = np.diff(angles_sorted, prepend=angles_sorted[-1] - 360, append=angles_sorted[0] + 360)



            # Q1 = np.percentile(diffs, 25)
            # Q3 = np.percentile(diffs, 75)
            # IQR = Q3 - Q1
            # # 找到异常值的索引
            # outliers_iqr = np.where((diffs > Q3 + 20 * IQR)&(diffs > 20))[0]
            # if len(outliers_iqr)>0 :
            #     # 删除异常值和对应的角度
            #     actual_indices_to_remove = sorted_indices[outliers_iqr]
            #     angles = np.delete(angles, actual_indices_to_remove, axis=0)
            #     angles_positive = np.where(angles < 0, angles + 360, angles)
            #     filtered_cluster_points = np.delete(cluster_points, actual_indices_to_remove, axis=0)
            #     print(f"----Outlier found angle {angles[[outliers_iqr]]}----")
            #     #重新拟合弧线
            #     xc, yc, r = fit_arc(filtered_cluster_points)
            #     angles = np.arctan2(cluster_points[:, 1] - yc, cluster_points[:, 0] - xc) * 180 / np.pi
            #     angles_positive = np.where(angles < 0, angles + 360, angles)



            threshold = 50  # 异常值阈值
            indices_to_remove = []
            outlier_indices = [i for i, diff in enumerate(diffs) if diff > threshold]
            if len(outlier_indices) > 1:
                indices_to_remove = [i for i in outlier_indices if i+1 in outlier_indices]
                outlier_indices = [i for i in outlier_indices if i not in indices_to_remove and i - 1 not in indices_to_remove and i + 1 not in indices_to_remove]
                # # 删除首尾
                # if len(outlier_indices) > 1:
                #     outlier_indices.remove(0) if 0 in outlier_indices else None
                #     outlier_indices.remove(len(diffs)-1) if len(diffs)-1 in outlier_indices else None
            
                # Step 4: 检查剩余点的区间范围
                while len(outlier_indices) > 1:
                    start, end = outlier_indices[0], outlier_indices[1]
                    if end - start <= 10:
                        indices_to_remove.extend(range(start, end))  # 添加区间范围内的点
                        outlier_indices = outlier_indices[2:] 
                    else:
                        outlier_indices.remove(start)
                print(f"indices_to_remove: {indices_to_remove}") if len(indices_to_remove) > 0 else None
    
            # 从 cluster_points 删除对应的点
            actual_indices_to_remove = sorted_indices[indices_to_remove]
            filtered_cluster_points = np.delete(cluster_points, actual_indices_to_remove, axis=0)   
            angles_positive = np.delete(angles_sorted, indices_to_remove, axis=0)

            # 从 hits 中删除对应的点
            hits.drop(hits.index[actual_indices_to_remove], inplace=True)
            # hits.reset_index(drop=True, inplace=True)

            if len(filtered_cluster_points)>10:
                # 重新拟合弧线
                xc, yc, r = fit_arc(filtered_cluster_points)
                angles = np.arctan2(cluster_points[:, 1] - yc, cluster_points[:, 0] - xc) * 180 / np.pi
                angles_positive = np.where(angles < 0, angles + 360, angles)

                #重新计算置信度
                confidence, _ = compute_confidence(filtered_cluster_points, xc, yc, r)
            
                theta1, theta2 = np.min(angles), np.max(angles)
                # 如果角度跨度超过 π（180°），表示需要跨越 0°，反转方向
                if theta2 - theta1  > 180 and (theta1 * theta2 < 0) and theta2 > 90 and theta1 < -90:   
                    if np.all((angles <= -90) | (angles >= 90)) or np.all(angles >= -90) or np.all((angles <= -90) | (angles >= 0)) or np.all((angles <= 0) | (angles >= 90)):  #全部在2、3象限 或 全部在1、2、4象限 或 全部在1、2、3象限 或全部在2、3、4象限
                        # 计算跨越 180° 时的角度范围
                        theta1, theta2 = np.min(angles_positive), np.max(angles_positive)
                
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
    cluster_name = 'ms' if cluster_method == 'meanshift' else 'db' if cluster_method == 'dbscan' else 'mct'
    finder_name = 'axs_fit_' if data_type == 'hit' else 'truth_fit_'
    folder_path = '/Users/Sevati/PycharmProjects/untitled/PID/Axs_Results_2/' + finder_name + cluster_name + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path + 'event' + str(evtCount) + '.jpg')
    plt.close()
    
    outfolder_path = '/Users/Sevati/PycharmProjects/untitled/PID/pid_data/MCTmcp_data/'   
    #将hits的字段“evtid,trkid,layer,wire,x,y,tx,ty,tz,rt,tdc,label”添加到已有txt文件
    savehits = hits[['evtid', 'trkid', 'layer', 'wire', 'x', 'y', 'tx', 'ty', 'tz', 'rt', 'tdc', 'label']]
    if os.path.exists(outfolder_path + 'hits_2.txt'):
        with open(outfolder_path + 'hits_2.txt', 'a') as f:
            savehits.to_csv(f, header=False, index=False)
    else:
        savehits.to_csv(outfolder_path + 'hits_2.txt', index=False)


# 为每个点分配权重
def assign_weights(event_df, vertical_layer_range, incline_layer_range):
    event_df['weight'] = event_df['layer'].apply(lambda x: 1.0 if x in vertical_layer_range else 0.6)
    return event_df
# 示例使用
if __name__ == "__main__":
    for evtCount in range (7,8):#(EvtNumTrain):
        print("-----------Processing event-----------:", evtCount)
        hits = get_hits(df_train, evtCount)
        coords = hits[['x', 'y']].values if data_type == 'hit' else hits[['tx', 'ty']].values
        # 直丝层和斜丝层的标号范围
        vertical_layer_range = list(range(8, 20)) + list(range(36, 43))
        incline_layer_range = list(range(0, 8)) + list(range(20, 36))
    
        # 为点分配权重
        hits = assign_weights(hits, vertical_layer_range, incline_layer_range)

        if len(coords) < 10:
            continue

        # 处理聚类并拟合圆弧
        hits, confidences = process_clusters(hits)   #置信度阈值
        
        # 可视化结果
        visualize_clusters(evtCount, hits, confidences)

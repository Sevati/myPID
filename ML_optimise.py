import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.optim as optim
import random
import torch
import torch.nn as nn
from math import sin,cos
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def set_seed(seed):
    # 1. 固定 PyTorch 的种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 如果使用GPU
    torch.backends.cudnn.deterministic = True  # 确保卷积等操作是确定性的
    torch.backends.cudnn.benchmark = False     # 禁用加速器的随机行为

    # 2. 固定 NumPy 的种子
    np.random.seed(seed)
    
    # 3. 固定 Python 自带的 random 模块的种子
    random.seed(seed)


def load_data(inputFile):
  # Load the data
  name=['evtid','trkid','layer','wire','x','y','rt','tdc']
  df=pd.read_table(inputFile,names=name,sep=',',header=None)
  return df


def get_hits(data, evtCount):
  df = data[data['evtid']==evtCount]
  raw = df.loc[(df['trkid'] > 0)]
#   raw = data.loc[(data['trkid'] > 0)]
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

    return results, success_rate


# 计算成功率
def calculate_success_rate(gt_labels, method_labels, out_successful_classes, out_total_classes):
    # 计算效率和纯度
    results, success_rate = calculate_efficiency_and_purity(gt_labels, method_labels)
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


# 计算调整兰德指数（Adjusted Rand Index）
def puredef(data_set, eps, min_samples, groundtruth_labels):
        X = pd.concat([data_set[df][['finalX', 'finalY']] for df in data_set], ignore_index=True).values
        cluster_out = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        cluster_labels = cluster_out.labels_

        cluster_labels = np.array(cluster_labels)
        groundtruth_labels = np.array(groundtruth_labels)

        # 确保输入是有效的
        if len(cluster_labels) != len(groundtruth_labels):
            raise ValueError("长度不匹配：cluster_labels 和 groundtruth_labels 必须有相同长度")

        # 创建混淆矩阵
        conf_matrix = confusion_matrix(groundtruth_labels, cluster_labels)

        # 使用匈牙利算法寻找最优标签对齐
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)

        # 计算经过优化后的ARI
        new_cluster_labels = np.zeros_like(cluster_labels)
        for i, j in zip(row_ind, col_ind):
            new_cluster_labels[cluster_labels == j] = i

        # 计算和返回调整兰德指数（Adjusted Rand Index）
        ari = adjusted_rand_score(groundtruth_labels, new_cluster_labels)
        return ari


# 1. 定义代理模型（简单的MLP神经网络）
class ParamModel(nn.Module):
    def __init__(self):
        super(ParamModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 2个输入：eps和min_samples
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # 预测 puredef 得分

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# 2. 定义评估函数，用当前参数训练 DBSCAN 并计算 puredef 得分
def evaluate_dbscan(data_set, eps, min_samples, X, y_true):
    eps = float(eps)
    min_samples = int(min_samples)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    groundtruth_labels = pd.concat([data_set[df]['trkid'] for df in data_set], ignore_index=True).values
    score = puredef(data_set, eps, min_samples, groundtruth_labels)
    return score


# 3. 定义训练函数，用 Adam 优化器训练 ParamModel
def train_model(Hits,X, y_true, epochs=100):
    model = ParamModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()  # 可以选择其他损失函数

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 随机生成 eps 和 min_samples（取值范围应根据你的数据调整）
        eps = torch.tensor(np.random.uniform(0.1, 2.0), requires_grad=True).unsqueeze(0)
        min_samples = torch.tensor(float(np.random.randint(3, 20)), requires_grad=True).unsqueeze(0)

        inputs = torch.cat([eps, min_samples], dim=0).float().unsqueeze(0)  # 组合输入
        pred_score = model(inputs)
 
        # 计算DBSCAN真实得分
        true_score = evaluate_dbscan(Hits, eps.item(), min_samples.item(), X, y_true)
        eps = float(eps)
        min_samples = int(min_samples)
        dbscan_labels = DBSCAN(eps=eps, min_samples=min_samples)
        # results, true_score = calculate_efficiency_and_purity(y_true, dbscan_labels)
        true_score = torch.tensor(true_score).float().unsqueeze(0) # 将true_score调整为[1, 1]
       
        # 计算损失
        loss = loss_fn(pred_score, true_score.unsqueeze(0))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    return model, losses


# 4. 可视化损失函数收敛情况
def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss during training")
    plt.show()
    plt.close()


# 5. 定义测试函数，用最优参数测试模型
def test_model(data_set, X, y_true, model):
    eps = model.fc1.weight.data[0][0].item()
    min_samples = int(model.fc1.weight.data[0][1].item())
    if min_samples < 2:
        min_samples = 2
    cluster_labels = DBSCAN(eps=eps, min_samples=4).fit_predict(X)
    # cluster_labels = db.labels_
    successful_classes = 0
    total_classes = 0
    success_rate, successful_classes, total_classes = calculate_success_rate(y_true, cluster_labels,successful_classes, total_classes)
    score = evaluate_dbscan(data_set, eps, min_samples, X, y_true)
    print(f"Test Score: {score:.2f}")
    
    '''
    # 可视化结果
    fig, axs = plt.subplots(1, 2, figsize=(20, 8)) 
    # Subplot 1: Groundtruth labels (x, y)
    unique_y_true = set(y_true)
    trkid_colors = cm.rainbow(np.linspace(0, 1, len(unique_y_true)))
    handles = []
    for k, col in zip(unique_y_true, trkid_colors):
        class_member_mask = (data_set['trkid'] == k)
        xy = data_set[class_member_mask]
        handle = axs[0].scatter(xy['x'], xy['y'], s=30, color=col, label=f'Cluster {k} ({np.sum(data_set["trkid"] == k)})',edgecolor='k')
        handles.append(handle)
    axs[0].set_title('Groundtruth Labels (x, y) - Event ' + str(7))
    axs[0].legend(handles=handles, loc='upper right')
    
    # Subplot 2: DBSCAN clustering results
    unique_dbscan_labels = set(cluster_labels)
    dbscan_colors = cm.rainbow(np.linspace(0, 1, len(unique_dbscan_labels)))
    handles = []
    for k, col in zip(unique_dbscan_labels, dbscan_colors):
        class_member_mask = (cluster_labels == k)
        xy = data_set[class_member_mask]
        handle = axs[1].scatter(xy['x'], xy['y'], s=30, color=col, label=f'Cluster {k} ({np.sum(cluster_labels == k)})',edgecolor='k')
        handles.append(handle)
    axs[1].set_title('DBSCAN Clustering Results - Event ' + str(7))
    axs[1].legend(handles=handles, loc='upper right')
    plt.show()
    plt.close()
    '''
    return success_rate


# 6. 定义主函数，用 GridSearchCV 寻找最优参数
def main():
    df = load_data("/Users/Sevati/PycharmProjects/untitled/PID/pid_data/2Ddata/hit_2.txt")
    EvtNum = 100
    Hits = {}
    for evtCount in range(EvtNum):
        Hits[evtCount] = get_hits(df,evtCount)

    # torch.manual_seed(3407)
    set_seed(3407)
    # 训练代理模型
    # X = Hits[:][['finalX', 'finalY']].values
    # y_true = Hits[:]['trkid'].values
    X = pd.concat([Hits[df][['finalX', 'finalY']] for df in Hits], ignore_index=True).values
    y_true = pd.concat([Hits[df]['trkid'] for df in Hits], ignore_index=True).values

    # 训练模型
    model, losses = train_model(Hits, X, y_true, epochs=100)

    # 可视化训练过程中的loss变化
    plot_losses(losses)
    test_Hits = {}
    test_Hits[0] = get_hits(df,5)
    test_Hits[1] = get_hits(df,10)
    test_X = pd.concat([test_Hits[df][['finalX', 'finalY']] for df in test_Hits], ignore_index=True).values
    test_y_true = pd.concat([test_Hits[df]['trkid'] for df in test_Hits], ignore_index=True).values

    # 测试模型
    score = test_model(test_Hits, test_X, test_y_true, model)
    print(f"success_rate: {score:.2f}")

if __name__ == '__main__':
  main()

#遗留的坑：
#训练模型时只用了一个示例，没有使用所有数据，没有使用交叉验证。
#loss函数选择Adjusted Rand Index而不是成功率。

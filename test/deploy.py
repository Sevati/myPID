import numpy as np
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler


class DBSCANClustering:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        return db.labels_

    def set_params(self, **params):
        self.eps = params.get('eps', self.eps)
        self.min_samples = params.get('min_samples', self.min_samples)

    def get_params(self):
        return {'eps': self.eps, 'min_samples': self.min_samples}
    

class MeanShiftClustering:
    def __init__(self, quantile=0.5, n_samples=10):
        self.quantile = quantile
        self.n_samples = n_samples

    def fit(self, X):
        bandwidth = estimate_bandwidth(X, quantile=self.quantile, n_samples=self.n_samples)
        db = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
        return db.labels_

    def set_params(self, **params):
        self.quantile = params.get('quantile', self.quantile)
        self.min_samples = params.get('n_samples', self.n_samples)

    def get_params(self):
        return {'quantile': self.quantile, 'n_samples': self.n_samples}


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


def calculate_efficiency_and_purity(gt_labels, dbscan_labels):
    unique_labels = np.unique(gt_labels)  # 包括所有标签，不过滤噪声
    results = {}

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

        results[label] = {'efficiency': efficiency, 'purity': purity}

    return results


def calculate_success_rate(gt_labels, method_labels, successful_classes, total_classes):
    # 计算效率和纯度
    results = calculate_efficiency_and_purity(gt_labels, method_labels)
    total_classes += len(np.unique(gt_labels))  # 包括所有真实类（噪声也算）

    for label, metrics in results.items():
        efficiency = metrics['efficiency']
        purity = metrics['purity']

        if efficiency > 0.8 and purity > 0.6:
            successful_classes += 1

    return successful_classes, total_classes


def custom_loss(gt_labels_list, pred_labels_list):
    successful_classes = 0
    total_classes = 0
    for gt_labels, pred_labels in zip(gt_labels_list, pred_labels_list):
        successful_classes, total_classes = calculate_success_rate(gt_labels, pred_labels, successful_classes, total_classes)
    success_rate = successful_classes / total_classes if total_classes > 0 else 0
    L_max = 100
    return L_max * (1 - success_rate)  # 成功率越大，loss越小


def simulated_annealing(train_data, train_labels, initial_eps, initial_min_samples, eps_range, min_samples_range, max_iterations, initial_temp, cooling_rate):
    clustering = DBSCANClustering(eps=initial_eps, min_samples=initial_min_samples)
    current_eps = initial_eps
    current_min_samples = initial_min_samples
    current_loss = float('inf')

    best_eps = current_eps
    best_min_samples = current_min_samples
    best_loss = current_loss

    temp = initial_temp

    for iteration in range(max_iterations):
        # 随机生成新的参数
        new_eps = random.uniform(eps_range[0], eps_range[1])
        new_min_samples = random.randint(min_samples_range[0], min_samples_range[1])
        
        # 设置新参数
        clustering.set_params(eps=new_eps, min_samples=new_min_samples)

        # 计算新参数下的损失
        all_pred_labels, all_truth_labels = [], []
        for i in range(len(train_data)):
            X = train_data[i]
            pred_labels = clustering.fit(X)
            pred_labels = reassign_labels(pred_labels)
            all_pred_labels.append(pred_labels)

            y = train_labels[i]
            y = reassign_labels(y)
            all_truth_labels.append(y)

        new_loss = custom_loss(all_truth_labels, all_pred_labels)

        # 接受新参数的概率
        acceptance_prob = np.exp((current_loss - new_loss) / temp)

        # 根据概率决定是否接受新参数
        if new_loss < current_loss or random.uniform(0, 1) < acceptance_prob:
            current_eps = new_eps
            current_min_samples = new_min_samples
            current_loss = new_loss

            if new_loss < best_loss:
                best_eps = new_eps
                best_min_samples = new_min_samples
                best_loss = new_loss
                
        # 降低温度
        temp *= cooling_rate

        print(f'Iteration {iteration+1}, Temp: {temp:.4f}, Current Loss: {current_loss:.4f}, Best Loss: {best_loss:.4f}')

    return best_eps, best_min_samples, best_loss



def train_dbscan(train_data, train_labels, initial_eps, initial_min_samples, learning_rate, steps_per_epoch, num_epochs, eps_range, min_samples_range):
    clustering = DBSCANClustering(eps=initial_eps, min_samples=initial_min_samples)
    eps = initial_eps
    min_samples = initial_min_samples
    loss_history = []

    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            all_pred_labels,all_truth_labels = [],[]
            for i in range(len(train_data)):
                X = train_data[i]
                pred_labels = clustering.fit(X)
                pred_labels = reassign_labels(pred_labels)
                all_pred_labels.append(pred_labels)

                y = train_labels[i]
                y = reassign_labels(y)
                all_truth_labels.append(y)
                # print(custom_loss(pred_labels,y))

            # Calculate loss
            loss = custom_loss(all_truth_labels, all_pred_labels)
            print(f'Epoch {epoch+1}, Step {step+1}, Loss: {loss}')
            loss_history.append(loss)

            # Update the parameters with constraints
            eps = max(eps_range[0], min(eps_range[1], eps - learning_rate * loss))
            min_samples = max(min_samples_range[0], min(min_samples_range[1], min_samples - int(learning_rate * loss)))
            clustering.set_params(eps=eps, min_samples=min_samples)
    
    return clustering, loss_history


def train_meanshift(train_data, train_labels, initial_eps, initial_n_samples, learning_rate, steps_per_epoch, num_epochs, eps_range, n_samples_range):
    clustering = MeanShiftClustering(quantile=initial_eps, n_samples=initial_n_samples)
    eps = initial_eps
    n_samples = initial_n_samples
    loss_history = []

    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            all_pred_labels,all_truth_labels = [],[]
            for i in range(len(train_data)):
                X = train_data[i]
                pred_labels = clustering.fit(X)
                pred_labels = reassign_labels(pred_labels)
                all_pred_labels.append(pred_labels)

                y = train_labels[i]
                y = reassign_labels(y)
                all_truth_labels.append(y)
                # print(custom_loss(pred_labels,y))

            # Calculate loss
            loss = custom_loss(all_truth_labels, all_pred_labels)
            print(f'Epoch {epoch+1}, Step {step+1}, Loss: {loss}')
            loss_history.append(loss)

            # Update the parameters with constraints
            eps = max(eps_range[0], min(eps_range[1], eps - learning_rate * loss))
            n_samples = max(n_samples_range[0], min(n_samples_range[1], n_samples - int(learning_rate * loss)))
            clustering.set_params(eps=eps, n_samples=n_samples)
    
    return clustering, loss_history


def visualize_loss(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()
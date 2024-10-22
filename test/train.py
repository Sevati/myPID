import pandas as pd
import numpy as np
from PID.ML.test.deploy1 import visualize_loss, custom_loss, reassign_labels, train_dbscan, train_meanshift
from math import cos, sin

def load_data(inputFile):
    # Load the data
    name = ['evtid', 'trkid', 'layer', 'wire', 'x', 'y', 'rt', 'tdc']
    df = pd.read_table(inputFile, names=name, sep=',', header=None)
    return df


def get_hits(data, evtCount):
    df = data[data['evtid'] == evtCount]
    raw = df.loc[(df['trkid'] > 0)]
    raw = raw.reset_index(drop=True)
    FinalX = []
    FinalY = []
    for i in list(raw.index):
        temp = raw['x'][i] * raw['x'][i] + raw['y'][i] * raw['y'][i]
        trans_x = 2 * raw['x'][i] / temp
        trans_y = 2 * raw['y'][i] / temp
        alpha = np.arctan2(trans_y, trans_x)
        FinalX.append(cos(alpha))
        FinalY.append(sin(alpha))
    raw['finalX'] = FinalX
    raw['finalY'] = FinalY
    return raw


def main():
    df_train = load_data("/Users/Sevati/PycharmProjects/untitled/PID/pid_data/2Ddata/hit_2.txt")    
    EvtNumTrain = 1000
    
    train_points = {}
    train_labels = {}
    
    for evtCount in range(EvtNumTrain):
        hits = get_hits(df_train, evtCount)
        train_points[evtCount] = hits[['finalX', 'finalY']].values
        train_labels[evtCount] = hits['trkid'].values

    # 设置初始参数
    initial_eps = 0.5
    initial_n_samples = 5
    learning_rate = 0.05
    steps_per_epoch = 10
    num_epochs = 5

    # 参数取值区间
    eps_range = (0.1, 1.0)
    n_samples_range = (1, 100)
    
    num_epochs = 5
    steps_per_epoch = 10
    learning_rate = 0.01

    # 训练模型
    dbscan_model, loss_history = train_dbscan(list(train_points.values()), list(train_labels.values()), initial_eps, initial_n_samples, learning_rate, steps_per_epoch, num_epochs, eps_range, n_samples_range)
    meanshift_model, loss_history = train_meanshift(list(train_points.values()), list(train_labels.values()), initial_eps, initial_n_samples, learning_rate, steps_per_epoch, num_epochs, eps_range, n_samples_range)

    # 可视化 loss
    visualize_loss(loss_history)


    # 使用测试集评估模型
    df_test = load_data("/Users/Sevati/PycharmProjects/untitled/PID/pid_data/2Ddata/hit_3.txt")
    EvtNumTest = 10

    test_points = {}
    test_labels = {}
    
    for evtCount in range(EvtNumTest):
        hits = get_hits(df_test, evtCount)
        test_points[evtCount] = hits[['finalX', 'finalY']].values
        test_labels[evtCount] = hits['trkid'].values

    all_pred_labels_test,all_truth_labels_test = [],[]
    for i in range(len(test_points)):
        X = test_points[i]
        pred_labels = dbscan_model.fit(X)
        pred_labels = reassign_labels(pred_labels)
        all_pred_labels_test.append(pred_labels)

        y = test_labels[i]
        y = reassign_labels(y)
        all_truth_labels_test.append(y)
    
    test_loss = custom_loss(all_truth_labels_test, all_pred_labels_test)
    print(f'Test Loss: {test_loss}')

    test_accuracy = 1 - test_loss / 100  
    print(f'Test success rate: {test_accuracy}')


if __name__ == "__main__":
    main()
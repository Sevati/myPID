import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from math import sin,cos

def load_data(inputFile):
  # Load the data
  name=['evtid','trkid','layer','wire','x','y','tx','ty','tz','rt','tdc']
  df=pd.read_table(inputFile,names=name,sep=',',header=None)
  return df


def get_hits(data,evtCount):
  df = data[data['evtid']==evtCount]
  raw = df.loc[(df['trkid'] > 0)]
  raw = raw.reset_index(drop=True)
  FinalX =[]
  FinalY =[]
  for i in list(raw.index):
     temp = raw['tx'][i]*raw['tx'][i] + raw['ty'][i]*raw['ty'][i]
     trans_x = 2*raw['tx'][i] / temp
     trans_y = 2*raw['ty'][i] / temp
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


def processing_data(inputFile):
    df = load_data(inputFile)
    # 对每个`evtid`的子集标准化特征值
    feature_list = []
    label_list = []
    max_len = 0
    EvtNum = 1000
    for evtCount in range(EvtNum):
        # print(f'processing event======================: {evtCount}')
        Hits = get_hits(df,evtCount) 
        if Hits.shape[0] < 10 :  
          continue
        data_set = [Hits.finalX,Hits.finalY]
        data_set = np.array(data_set).T 

        X = Hits[['layer', 'wire', 'x', 'y', 'tx','ty','tz', 'rt', 'tdc']].values #点的个数
        y = Hits['trkid'].values  #点对应的径迹
        y = reassign_labels(y)
        scaler = StandardScaler()           # 标准化特征值
        X_scaled = scaler.fit_transform(X)  # 将处理后的特征值和标签值分别添加到特征列表和标签列表中  
        feature_list.append(X_scaled)
        label_list.append(y)
        if X.shape[0] > max_len:
            max_len = X.shape[0]  # 找出最长的序列长度
    
    max_len_index = max(range(len(feature_list)), key=lambda i: len(feature_list[i]))
    max_len = len(feature_list[max_len_index])

    print("最长序列的索引:", max_len_index)
    print("最长序列的长度:", max_len)

    min_len_index = min(range(len(feature_list)), key=lambda i: len(feature_list[i]))
    min_len = len(feature_list[min_len_index])

    print("最短序列的索引:", min_len_index)
    print("最短序列的长度:", min_len)

    # 进行填充，使所有序列长度一致
    padded_features = []
    padded_labels = []
    for i in range(len(feature_list)):
        padded_X = np.pad(feature_list[i], ((0, max_len - feature_list[i].shape[0]), (0, 0)), 'constant')
        padded_y = np.pad(label_list[i], (0, len(label_list[i])), 'constant')
        padded_features.append(padded_X)
        padded_labels.append(padded_y)

    X = np.array(padded_features)
    y = np.array(padded_labels)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# def main():
#     # 数据预处理
#     df = processing_data(file_path)
#     X_train, X_test, y_train, y_test = load_data(file_path)


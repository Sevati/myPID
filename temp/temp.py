#一个计算效率和纯度的demo
# '''
from collections import defaultdict
import pandas as pd

def compute_efficiency_and_purity(data, gt_labels, db_labels, label_name,efficiency_list,purity_list):
    def get_indices(labels):
        indices = defaultdict(list)
        for index, label in enumerate(labels):
            indices[label].append(index)
        return indices

    # 获得每个类别的索引
    true_indices = get_indices(gt_labels)
    cluster_indices = get_indices(db_labels)

    efficiencies = {}
    purities = {}

    # 建立真值类别和聚类结果类别之间的对应关系
    true_to_cluster_map = {}

    # 计算真值中的每个类别在聚类结果中的最大匹配类别
    for true_class, gt_indices in true_indices.items():
        if len(gt_indices) < 5:  # 如果某类别点数少于3，则跳过
            continue
        cluster_count = defaultdict(int)
        for idx in gt_indices:
            cluster_count[db_labels[idx]] += 1
        mapped_cluster = max(cluster_count, key=cluster_count.get)
        true_to_cluster_map[true_class] = mapped_cluster

    # 计算每个类别的效率和纯度
    for true_class, mapped_cluster in true_to_cluster_map.items():
        gt_indices = true_indices[true_class]
        db_indices = cluster_indices[mapped_cluster]
        
        correct_count_efficiency = sum(1 for idx in gt_indices if db_labels[idx] == mapped_cluster)
        correct_count_purity = sum(1 for idx in db_indices if gt_labels[idx] == true_class)
        
        efficiency = correct_count_efficiency / len(gt_indices) if len(gt_indices) > 0 else 0
        purity = correct_count_purity / len(db_indices) if len(db_indices) > 0 else 0
         
        efficiency_list.append(efficiency)
        purity_list.append(purity)

        for idx in gt_indices:
            efficiencies[idx] = efficiency
            purities[idx] = purity
    
    data[f'{label_name}_efficiency'] = efficiencies
    data[f'{label_name}_purity'] = purities
    return efficiencies, purities

# 真值和聚类结果
true_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
cluster_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
label_name = 'test'

# 将数据封装在DataFrame中
data = {
    'true_labels': true_labels,
    'cluster_labels': cluster_labels
}

df = pd.DataFrame(data)
efficiency_list,purity_list = [],[]

efficiencies, purities = compute_efficiency_and_purity(df, true_labels, cluster_labels,label_name,efficiency_list,purity_list)
print(df)
# '''


#横向拼接两个文件夹中的同名文件
'''
import os
from PIL import Image

# 设置路径
layer_path = '/Users/Sevati/PycharmProjects/untitled/PID/axs_layer'
mct_path = '/Users/Sevati/PycharmProjects/untitled/PID/axs_mct'
output_path = '/Users/Sevati/PycharmProjects/untitled/PID/axs'

# 确保输出目录存在
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 获取axs_layer文件夹中的所有文件名
layer_files = os.listdir(layer_path)
mct_files = os.listdir(mct_path)

# 确保两个文件夹中的文件名一致
common_files = set(layer_files) & set(mct_files)

# 遍历公共文件名
for filename in common_files:
    layer_img_path = os.path.join(layer_path, filename)
    mct_img_path = os.path.join(mct_path, filename)
    
    # 打开图片
    layer_img = Image.open(layer_img_path)
    mct_img = Image.open(mct_img_path)
    
    # 确保两个图片大小相同
    if layer_img.size != mct_img.size:
        print(f"Skipping {filename}, as the image sizes do not match.")
        continue
    
    # 创建新的图像，大小为两个图像横向拼接
    new_img = Image.new('RGB', (layer_img.width + mct_img.width, layer_img.height))
    
    # 粘贴图像
    new_img.paste(layer_img, (0, 0))
    new_img.paste(mct_img, (layer_img.width, 0))
    
    # 保存拼接后的图像
    new_img.save(os.path.join(output_path, filename))

print("All images have been concatenated and saved.")
# '''
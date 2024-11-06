import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取数据
file_path = "/Users/Sevati/PycharmProjects/untitled/PID/wireXY.txt"
data = pd.read_csv(file_path, sep=',', header=None)
data.columns = ['Index', 'Layer', 'Wire', 'X', 'Y', 'Group']

# 定义超层类型
straight_layers = [2, 3, 4, 9, 10]  # 直丝超层
reverse_slant_layers = [0, 5, 7]  # 逆向的斜丝超层
positive_slant_layers = [1, 6, 8]  # 正向的斜丝超层

# 创建一个层级偏移特征
data['LayerType'] = data['Layer'] % 2  # 奇数层和偶数层
data['Offset'] = data['LayerType'] * 0.5  # 根据奇偶层设置偏移
# data['Offset'] = np.where(data['Layer'] % 2 == 0, 0, 0.5)

# 将数据按超层类型分类，并重新索引
data_straight = data[data['Group'].isin(straight_layers)].reset_index(drop=True)
data_reverse_slant = data[data['Group'].isin(reverse_slant_layers)].reset_index(drop=True)
data_positive_slant = data[data['Group'].isin(positive_slant_layers)].reset_index(drop=True)

# 提取特征和目标变量
def prepare_data(data):
    X = data[['Layer', 'Wire', 'Offset']]
    y_X = data['X']
    y_Y = data['Y']
    return X, y_X, y_Y

X_straight, y_X_straight, y_Y_straight = prepare_data(data_straight)
X_reverse_slant, y_X_reverse_slant, y_Y_reverse_slant = prepare_data(data_reverse_slant)
X_positive_slant, y_X_positive_slant, y_Y_positive_slant = prepare_data(data_positive_slant)

# 定义一个函数来训练和评估模型
def train_and_evaluate(X, y_X, y_Y):
    # 分割数据集
    X_train, X_test, y_train_X, y_test_X = train_test_split(X, y_X, test_size=0.2, random_state=42)
    X_train, X_test, y_train_Y, y_test_Y = train_test_split(X, y_Y, test_size=0.2, random_state=42)

    # 构建和训练模型（随机森林回归）
    model_X = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model_X.fit(X_train, y_train_X)

    model_Y = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model_Y.fit(X_train, y_train_Y)

    # 进行预测
    y_pred_X = model_X.predict(X_test)
    y_pred_Y = model_Y.predict(X_test)

    # 计算误差
    mse_X = mean_squared_error(y_test_X, y_pred_X)
    mae_X = mean_absolute_error(y_test_X, y_pred_X)
    mse_Y = mean_squared_error(y_test_Y, y_pred_Y)
    mae_Y = mean_absolute_error(y_test_Y, y_pred_Y)
    return y_test_X.reset_index(drop=True), y_test_Y.reset_index(drop=True), y_pred_X, y_pred_Y, mse_X, mae_X, mse_Y, mae_Y


# 训练并评估模型
y_test_X_straight, y_test_Y_straight, y_pred_X_straight, y_pred_Y_straight, mse_X_straight, mae_X_straight, mse_Y_straight, mae_Y_straight = train_and_evaluate(X_straight, y_X_straight, y_Y_straight)
y_test_X_reverse_slant, y_test_Y_reverse_slant, y_pred_X_reverse_slant, y_pred_Y_reverse_slant, mse_X_reverse_slant, mae_X_reverse_slant, mse_Y_reverse_slant, mae_Y_reverse_slant = train_and_evaluate(X_reverse_slant, y_X_reverse_slant, y_Y_reverse_slant)
y_test_X_positive_slant, y_test_Y_positive_slant, y_pred_X_positive_slant, y_pred_Y_positive_slant, mse_X_positive_slant, mae_X_positive_slant, mse_Y_positive_slant, mae_Y_positive_slant = train_and_evaluate(X_positive_slant, y_X_positive_slant, y_Y_positive_slant)

# 输出误差
print("直丝超层 - X坐标 - 均方误差 (MSE):", mse_X_straight, ", 平均绝对误差 (MAE):", mae_X_straight)
print("直丝超层 - Y坐标 - 均方误差 (MSE):", mse_Y_straight, ", 平均绝对误差 (MAE):", mae_Y_straight)
print("逆向斜丝超层 - X坐标 - 均方误差 (MSE):", mse_X_reverse_slant, ", 平均绝对误差 (MAE):", mae_X_reverse_slant)
print("逆向斜丝超层 - Y坐标 - 均方误差 (MSE):", mse_Y_reverse_slant, ", 平均绝对误差 (MAE):", mae_Y_reverse_slant)
print("正向斜丝超层 - X坐标 - 均方误差 (MSE):", mse_X_positive_slant, ", 平均绝对误差 (MAE):", mae_X_positive_slant)
print("正向斜丝超层 - Y坐标 - 均方误差 (MSE):", mse_Y_positive_slant, ", 平均绝对误差 (MAE):", mae_Y_positive_slant)

# 可视化结果
def plot_predictions(y_test_X, y_pred_X, y_test_Y, y_pred_Y, title):
    plt.figure(figsize=(10, 10))
    first = True  # 用于给散点添加标签
    for true_x, pred_x, true_y, pred_y in zip(y_test_X, y_pred_X, y_test_Y, y_pred_Y):
        plt.scatter(true_x, true_y, color='red', s=10, label='True' if first else "")
        plt.scatter(pred_x, pred_y, color='blue', s=10, label='Predicted' if first else "")
        plt.plot([true_x, pred_x], [true_y, pred_y], color='black', linestyle='--', linewidth=0.5)
        first = False  # 之后不再添加标签
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()

# 可视化预测结果
plot_predictions(y_test_X_straight, y_pred_X_straight, y_test_Y_straight, y_pred_Y_straight, 'Straight Layers Prediction')
plot_predictions(y_test_X_reverse_slant, y_pred_X_reverse_slant, y_test_Y_reverse_slant, y_pred_Y_reverse_slant, 'Reverse Slant Layers Prediction')
plot_predictions(y_test_X_positive_slant, y_pred_X_positive_slant, y_test_Y_positive_slant, y_pred_Y_positive_slant, 'Positive Slant Layers Prediction')
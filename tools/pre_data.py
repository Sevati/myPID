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

# 创建特征
data['LayerType'] = data['Layer'] % 2  # 奇数层和偶数层
data['Offset'] = data['LayerType'] * 0.5  # 根据奇偶层设置偏移

# 提取特征和目标变量
X = data[['Layer', 'Wire', 'Offset']]
y_X = data['X']
y_Y = data['Y']

# 数据集划分
X_train, X_test, y_train_X, y_test_X = train_test_split(X, y_X, test_size=0.2, random_state=42)
X_train, X_test, y_train_Y, y_test_Y = train_test_split(X, y_Y, test_size=0.2, random_state=42)

'''
# 使用多项式回归模型
poly_features = PolynomialFeatures(degree=2, include_bias=False)
model_X = Pipeline([("poly_features", poly_features), ("linear_regression", LinearRegression())])
model_X.fit(X_train, y_train_X)

model_Y = Pipeline([("poly_features", poly_features), ("linear_regression", LinearRegression())])
model_Y.fit(X_train, y_train_Y)
'''

# '''
# 构建和训练模型（随机森林回归）
model_X = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model_X.fit(X_train, y_train_X)

model_Y = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model_Y.fit(X_train, y_train_Y)
# '''

# 预测
y_pred_X = model_X.predict(X_test)
y_pred_Y = model_Y.predict(X_test)

# 计算误差
mse_X = mean_squared_error(y_test_X, y_pred_X)
mae_X = mean_absolute_error(y_test_X, y_pred_X)

mse_Y = mean_squared_error(y_test_Y, y_pred_Y)
mae_Y = mean_absolute_error(y_test_Y, y_pred_Y)

# 打印误差
print(f'Mean Squared Error for X: {mse_X}')
print(f'Mean Absolute Error for X: {mae_X}')
print(f'Mean Squared Error for Y: {mse_Y}')
print(f'Mean Absolute Error for Y: {mae_Y}')

# 可视化结果
'''
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test_X, y_pred_X, alpha=0.3)
plt.xlabel('True X')
plt.ylabel('Predicted X')
plt.title('X Prediction')

plt.subplot(1,2,2)
plt.scatter(y_test_Y, y_pred_Y, alpha=0.3)
plt.xlabel('True Y')
plt.ylabel('Predicted Y')
plt.title('Y Prediction')
plt.show()
'''
plt.figure(figsize=(12, 12))

# 对于每个测试样本
for i in range(len(y_test_X)):
    # 真实值和预测值的X坐标
    true_x = y_test_X.iloc[i]
    pred_x = y_pred_X[i]
    
    # 真实值和预测值的Y坐标
    true_y = y_test_Y.iloc[i]
    pred_y = y_pred_Y[i]
    
    # 绘制真实值的红点和预测值的蓝点
    plt.scatter(true_x, true_y, color='red', s=10, label='True' if i == 0 else "")
    plt.scatter(pred_x, pred_y, color='blue', s=10, label='Predicted' if i == 0 else "")
    
    # 用黑色的细虚线连接真实值和预测值
    plt.plot([true_x, pred_x], [true_y, pred_y], color='black', linestyle='--', linewidth=0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('X and Y Prediction')
plt.legend()
plt.show()
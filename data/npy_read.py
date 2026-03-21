import numpy as np

# 读取 npy 文件
data = np.load("math1_Q_matrix.npy")

# 打印数据
print(data)
print(len(data[0]))
# 查看数据类型
print(type(data))

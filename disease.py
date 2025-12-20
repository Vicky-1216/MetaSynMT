# import numpy as np
#
# # 读取.npy文件
# data = np.load('D:/daima/Muthene-main/Muthene_dataset/common files/cellline_expression.npy', allow_pickle=True)
#
# # 查看数据
# print(data)
#
# import numpy as np
# import pandas as pd
#
# # 读取.npy文件
# data = np.load('D:/daima/Muthene-main/Muthene_dataset/common files/cellline_expression.npy', allow_pickle=True)
#
# # 将数据转换为DataFrame
# df = pd.DataFrame(data)
#
# # 将DataFrame保存为.csv文件
# df.to_csv('D:/daima/Muthene-main/Muthene_dataset/common files/cellline_expression.csv', index=False)
# import pandas as pd
#
# # 读取 gene.csv 和 expression.csv 文件的数据
# gene_df = pd.read_csv('D:/daima/Muthene-main/echino_dataset/gene.csv', header=None)
# expression_df = pd.read_csv('D:/daima/Muthene-main/echino_dataset/expression.csv')
#
# # 遍历 expression.csv 的第一行，找到与 gene.csv 第一行相同的单元格，并将 gene.csv 第二行的值赋给下方单元格
# for col in expression_df.columns:
#     if col in gene_df.iloc[0].values:
#         gene_col_index = gene_df.iloc[0][gene_df.iloc[0] == col].index[0]
#         expression_df.loc[1, col] = gene_df.iloc[1, gene_col_index]
#
# # 保存修改后的 expression.csv 文件为 expression1.csv
# expression_df.to_csv('D:/daima/Muthene-main/echino_dataset/expression1.csv', index=False)
#
#
# #new_df.to_csv('D:/daima/Muthene-main/echino_dataset/expression.csv')
#
# import numpy as np
#
# # 1. 读取 expression_reduced2.npy 文件
# data = np.load('C:/users/Administrator/Desktop/Muthene-main/echino_dataset/expression_reduced4.npy', allow_pickle=True)
# data = data.astype(float)
# #2. 计算数据的均值和标准差
# mean = np.mean(data, axis=0)
# std = np.std(data, axis=0)
#
# # 3. 对数据进行标准化
# normalized_data = (data - mean) / std
#
# # 4. 保存标准化后的数据到 expression_reduced_normalized2.npy 文件
# np.save('C:/users/Administrator/Desktop/Muthene-main/echino_dataset/expression_reduced_normalized4.npy', normalized_data)

# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('D:/daima/Muthene-main/echino_dataset/expression4副.csv')
#
# # 将所有非数值的值替换为 NaN（或一个适当的数值），然后转换为数值类型
# df = df.replace('', pd.NA)  # 假设空字符串''表示缺失值
# df = df.astype(float)  # 将所有列转换为浮点数类型
#
# # 将所有大于0的值替换为1
# df_replaced = df.applymap(lambda x: 1 if x > 0 else x)
#
# # 保存到新的CSV文件
# df_replaced.to_csv('D:/daima/Muthene-main/echino_dataset/expression4.csv', index=False)

#
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#
# 加载数据
data = np.load('C:/users/Administrator/Desktop/Muthene-main/echino_dataset/disease_features.npy', allow_pickle=True)
print(data)
#初始化MinMaxScaler对象
scaler = MinMaxScaler()

# 对数据进行归一化
normalized_data = scaler.fit_transform(data)
np.set_printoptions(threshold=np.inf)

print(normalized_data)
# 将归一化后的数据保存到.npy文件中
np.save('C:/users/Administrator/Desktop/Muthene-main/echino_dataset/disease_features_normalized.npy', normalized_data)




import pandas as pd

# 读取CSV文件

# import csv
# import numpy as np
# # 1. 读取1.csv文件
# data = []
# with open('C:/users/Administrator/Desktop/Muthene-main/echino_dataset/expression_reduced3.csv', 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     next(csvreader)
#     for row in csvreader:
#         data.append([float(cell) for cell in row])
#
# # 2. 计算疾病之间的Jaccard相似性
# def jaccard_similarity(row1, row2):
#     union = sum(max(cell1, cell2) for cell1, cell2 in zip(row1, row2))
#     intersect = sum(min(cell1, cell2) for cell1, cell2 in zip(row1, row2))
#     return intersect / union if union != 0 else 0
#
# # 计算相似性矩阵
# similarity_matrix = [[jaccard_similarity(row1, row2) for row2 in data] for row1 in data]
#
# # 3. 将相似性保存到2.csv文件
# # with open('C:/users/Administrator/Desktop/Muthene-main/echino_dataset/similar.csv', 'w', newline='') as csvfile:
# #     csvwriter = csv.writer(csvfile)
# #     for row in similarity_matrix:
# #         csvwriter.writerow(row)
#
# similarity_array = np.array(similarity_matrix, dtype=np.float32)  # 假设数据是浮点数
#
# # 保存NumPy数组为.npy文件
# np.save('C:/users/Administrator/Desktop/Muthene-main/echino_dataset/similar.npy', similarity_array)
#

# import numpy as np
# import pandas as pd
#
# # 步骤1: 读取CSV文件
# # 假设CSV文件中的数据是逗号分隔的，并且没有标题行
# # 如果有标题行，可以设置 header=None 并使用 names 参数指定列名
# df = pd.read_csv('C:/users/Administrator/Desktop/Muthene-main/echino_dataset/disease_features.csv')
#
# # 步骤2: 将CSV数据转换为NumPy数组
# # 假设CSV文件中的数据已经是数值类型，可以直接转换
# # 如果CSV文件中包含非数值类型数据，需要先进行适当的转换
# data = df.to_numpy()
#
# # 步骤3: 将NumPy数组保存为.npy文件
# np.save('C:/users/Administrator/Desktop/Muthene-main/echino_dataset/disease_features.npy', data)
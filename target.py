
import pandas as pd

def drug_target_mapping(drug_target_file_path, test_file_path):
    drug_target_df = pd.read_csv(drug_target_file_path, header=None, skiprows=1)
    drug_target_dict = {}
    for index, row in drug_target_df.iterrows():
        drug = row[0]
        target = row[1]
        if drug in drug_target_dict:
            drug_target_dict[drug].append(target)
        else:
            drug_target_dict[drug] = [target]

    test_df = pd.read_csv(test_file_path, header=None, skiprows=1)

    drug_list_1 = test_df[0].tolist()
    drug_list_2 = test_df[1].tolist()

    unique_drugs = set(drug_list_1) | set(drug_list_2)

    result = {}

    for drug in unique_drugs:
        if drug in drug_target_dict:
            result[drug] = drug_target_dict[drug]
        else:
            result[drug] = []

    return result

drug_target_file_path = './static/drug-target.csv'
test_file_path = './static/test_drug_drug_samples_name_top.csv'
result_mapping = drug_target_mapping(drug_target_file_path, test_file_path)
for drug, targets in result_mapping.items():
    print(f"drug: {drug}, target: {', '.join(targets) if targets else 'no targets'}")



#
# import pandas as pd
#
# # 读取drug-target.csv文件
# drug_target_df = pd.read_csv('./static/drug-target.csv')
#
# # 将靶标名分割为列表，并创建字典
# drug_target_dict = {
#     drug: target.split(',') for drug, target in zip(drug_target_df['drug'], drug_target_df['target'])
# }
#
# # 读取test.csv文件
# test_df = pd.read_csv('./static/test_drug_drug_samples_name_top.csv')
#
# # 获取test.csv中第一列和第二列的药物名，并取并集
# unique_drug_names = set(test_df.iloc[:, 0]).union(set(test_df.iloc[:, 1]))
#
# # 定义一个函数，用于查找药物对应的靶标列表
# def find_targets(drug_name, drug_target_dict):
#     return drug_target_dict.get(drug_name, [])
#
# # 创建一个空的DataFrame来存储结果
# result_df = pd.DataFrame(columns=['Drug', 'Targets'])
#
# result_rows = []  # 创建一个空列表来存储每一行的数据
# for drug_name in unique_drug_names:
#     targets = find_targets(drug_name, drug_target_dict)
#     result_rows.append({'Drug': drug_name, 'Targets': targets})
#
# # 使用concat将列表转换为DataFrame
# result_df = pd.concat([pd.DataFrame(result_rows)], ignore_index=True)
#
# # 重置索引并丢弃旧索引
# result_df.reset_index(drop=True, inplace=True)
#
# # 显示结果
# print(result_df)
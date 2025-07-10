
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




# import pandas as pd
#
# drug_target_df = pd.read_csv('./static/drug-target.csv')
#
# drug_target_dict = {
#     drug: target.split(',') for drug, target in zip(drug_target_df['drug'], drug_target_df['target'])
# }
#
# test_df = pd.read_csv('./static/test_drug_drug_samples_name_top.csv')
#
# unique_drug_names = set(test_df.iloc[:, 0]).union(set(test_df.iloc[:, 1]))
#
# def find_targets(drug_name, drug_target_dict):
#     return drug_target_dict.get(drug_name, [])
#
# result_df = pd.DataFrame(columns=['Drug', 'Targets'])
#
# for drug_name in unique_drug_names:
#     targets = find_targets(drug_name, drug_target_dict)
#     result_rows.append({'Drug': drug_name, 'Targets': targets})
#
# result_df = pd.concat([pd.DataFrame(result_rows)], ignore_index=True)
#
# result_df.reset_index(drop=True, inplace=True)
#
# print(result_df)

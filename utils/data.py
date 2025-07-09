import numpy as np
import scipy
import pickle
import pandas as pd
from dgl.data import DGLDataset
import torch
import dgl
import os


#
def load_data_te(prefix): # your folder to store the required original data ='C:/Users/Administrator/Desktop/Muthene-main/echino_dataset/fold4/'
    #print('the path of source file is :', prefix)
    # read drug adjlist files using relative index 四种不同元路径的邻接列表数据
    in_file = open(prefix + '0/0-1-0.adjlist', 'r')#0-1-0是元路径
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '0/0-1-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '0/0-1-1-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    # in_file = open(prefix + '0/0-1-1-1-1-0.adjlist', 'r')
    # adjlist03 = [line.strip() for line in in_file]
    # adjlist03 = adjlist03
    # in_file.close()
    in_file = open(prefix + '0/0-te-0.adjlist', 'r')
    adjlist04 = [line.strip() for line in in_file]
    adjlist04 = adjlist04
    in_file.close()

    # read the metapath instance files (stored as absolute index)
    in_file = open(prefix + '0/0-1-0_idx.pickle', 'rb')#对应元路径的索引
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '0/0-1-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '0/0-1-1-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    # in_file = open(prefix + '0/0-1-1-1-1-0_idx.pickle', 'rb')
    # idx03 = pickle.load(in_file)
    # in_file.close()
    in_file = open(prefix + '0/0-te-0_idx.pickle', 'rb')
    idx04 = pickle.load(in_file)
    in_file.close()

    # read adjacency matrix storing the all required interactions required by training读取邻接矩阵，存储训练所需的所有交互
    adjM = scipy.sparse.load_npz(prefix + 'adjM.npz')#存储了训练所需所有相互作用的邻接矩阵数据 adjM
    # type_mask is a mask storing all nodes' types
    type_mask = np.load(prefix + 'node_types.npy') #节点类型的掩码数据

    in_file = open(prefix + 'drug2absid_dict.pickle', 'rb')#药物到 ID 的映射字典。
    drug2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'target2absid_dict.pickle', 'rb')#靶标到 ID 的映射字典。
    target2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'disease2absid_dict.pickle', 'rb')#细胞系到 ID 的映射字典。
    disease2id_dict = pickle.load(in_file)
    in_file.close()

    train_val_test_drug_drug_samples = np.load(prefix + 'train_val_test_drug_drug_samples.npz')#训练、验证和测试集的药物-药物样本数据。
    train_val_test_drug_drug_labels = np.load(prefix + 'train_val_test_drug_drug_labels.npz')#训练、验证和测试集的药物-药物标签数据。

    if os.path.exists(prefix + 'expression_reduced_normalized3.npy'):
        disease_gene = np.load(prefix + 'expression_reduced_normalized3.npy', allow_pickle=True)
    else:
        disease_gene = np.zeros((len(disease2id_dict), 512))
        print('an empty disease data is generated.')

    if os.path.exists(prefix + 'similar.npy'):
        similar = np.load(prefix + 'similar.npy', allow_pickle=True)
    else:
        similar = np.zeros((len(disease2id_dict), 512))
        print('an empty disease data is generated.')

    return [[adjlist00, adjlist01, adjlist02, adjlist04],[adjlist00, adjlist01, adjlist02, adjlist04]], [[idx00, idx01, idx02, idx04],[idx00, idx01, idx02, idx04]], \
           adjM, type_mask, \
           [drug2id_dict, target2id_dict, disease2id_dict], \
           train_val_test_drug_drug_samples, train_val_test_drug_drug_labels , disease_gene, similar

#数据包括邻接列表、元路径实例、邻接矩阵、节点类型掩码、实体映射字典、药物-药物样本和标签
def load_data_se(prefix): # your folder to store the required original data ='C:/Users/Administrator/Desktop/Muthene-main/echino_dataset/side/fold4/'
    #print('the path of source file is :', prefix)

    # read drug adjlist files using relative index 四种不同元路径的邻接列表数据
    in_file = open(prefix + '0/0-1-0.adjlist', 'r')#0-1-0是元路径
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '0/0-1-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '0/0-1-1-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    # in_file = open(prefix + '0/0-1-1-1-1-0.adjlist', 'r')
    # adjlist03 = [line.strip() for line in in_file]
    # adjlist03 = adjlist03
    # in_file.close()
    in_file = open(prefix + '0/0-se-0.adjlist', 'r')
    adjlist04 = [line.strip() for line in in_file]
    adjlist04 = adjlist04
    in_file.close()

    # read the metapath instance files (stored as absolute index)
    in_file = open(prefix + '0/0-1-0_idx.pickle', 'rb')#对应元路径的索引
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '0/0-1-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '0/0-1-1-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    # in_file = open(prefix + '0/0-1-1-1-1-0_idx.pickle', 'rb')
    # idx03 = pickle.load(in_file)
    # in_file.close()
    in_file = open(prefix + '0/0-se-0_idx.pickle', 'rb')
    idx04 = pickle.load(in_file)
    in_file.close()

    # read adjacency matrix storing the all required interactions required by training读取邻接矩阵，存储训练所需的所有交互
    adjM = scipy.sparse.load_npz(prefix + 'adjM.npz')#存储了训练所需所有相互作用的邻接矩阵数据 adjM
    # type_mask is a mask storing all nodes' types
    type_mask = np.load(prefix + 'node_types.npy') #节点类型的掩码数据

    in_file = open(prefix + 'drug2absid_dict.pickle', 'rb')#药物到 ID 的映射字典。
    drug2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'target2absid_dict.pickle', 'rb')#靶标到 ID 的映射字典。
    target2id_dict = pickle.load(in_file)
    in_file.close()
    # in_file = open(prefix + 'disease2absid_dict.pickle', 'rb')#细胞系到 ID 的映射字典。
    # disease2id_dict = pickle.load(in_file)
    # in_file.close()

    train_val_test_drug_drug_samples = np.load(prefix + 'train_val_test_drug_drug_samples.npz')#训练、验证和测试集的药物-药物样本数据。
    train_val_test_drug_drug_labels = np.load(prefix + 'train_val_test_drug_drug_labels.npz')#训练、验证和测试集的药物-药物标签数据。

    return [[adjlist00, adjlist01, adjlist02, adjlist04],[adjlist00, adjlist01, adjlist02, adjlist04]], [[idx00, idx01, idx02, idx04],[idx00, idx01, idx02, idx04]], \
           adjM, type_mask, \
           [drug2id_dict, target2id_dict], \
           train_val_test_drug_drug_samples, train_val_test_drug_drug_labels
#TE模块 需要药物 细胞系和AE DNN用的

def load_HNEMA_DDI_data_te2(prefix='D:/daima/Muthene-main/echino_dataset/fold1/'): # your folder to store the required original data
    print('the path of source file is :', prefix)
    # read drug adjlist files using relative index 四种不同元路径的邻接列表数据
    in_file = open(prefix + '0/0-1-0.adjlist', 'r')#0-1-0是元路径
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '0/0-1-1-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + '0/0-1-1-1-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + '0/0-te-0.adjlist', 'r')
    adjlist03 = [line.strip() for line in in_file]
    adjlist03 = adjlist03
    in_file.close()

    # read the metapath instance files (stored as absolute index)
    in_file = open(prefix + '0/0-1-0_idx.pickle', 'rb')#对应元路径的索引
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '0/0-1-1-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '0/0-1-1-1-0_idx.pickle', 'rb')
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '0/0-te-0_idx.pickle', 'rb')
    idx03 = pickle.load(in_file)
    in_file.close()

    # read adjacency matrix storing the all required interactions required by training读取邻接矩阵，存储训练所需的所有交互
    adjM = scipy.sparse.load_npz(prefix + 'adjM.npz')#存储了训练所需所有相互作用的邻接矩阵数据 adjM
    # type_mask is a mask storing all nodes' types
    type_mask = np.load(prefix + 'node_types.npy') #节点类型的掩码数据

    in_file = open(prefix + 'drug2absid_dict.pickle', 'rb')#药物到 ID 的映射字典。
    drug2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'target2absid_dict.pickle', 'rb')#靶标到 ID 的映射字典。
    target2id_dict = pickle.load(in_file)
    in_file.close()
##    in_file = open(prefix + 'celline2relid_dict.pickle', 'rb')
##    celline2relid_dict = pickle.load(in_file)
##    in_file.close()
    in_file = open(prefix + 'disease2absid_dict.pickle', 'rb')#细胞系到 ID 的映射字典。
    disease2id_dict = pickle.load(in_file)
    in_file.close()
##    in_file = open(prefix + 'se_symbol2id_dict.pickle', 'rb')#副作用名称到 ID 的映射字典。
##    se_symbol2id_dict = pickle.load(in_file)

    train_val_test_drug_drug_samples = np.load(prefix + 'train_val_test_drug_drug_samples.npz')#训练、验证和测试集的药物-药物样本数据。
    train_val_test_drug_drug_labels = np.load(prefix + 'train_val_test_drug_drug_labels.npz')#训练、验证和测试集的药物-药物标签数据。

    if os.path.exists(prefix + 'expression_reduced_normalized.npy'):
        cellline_expression = np.load(prefix + 'expression_reduced_normalized.npy', allow_pickle=True)
    else:
        cellline_expression = np.zeros((len(disease2id_dict), 1024))
        print('an empty cell line expression data is generated.')

    return [[adjlist00, adjlist01, adjlist02, adjlist03],[adjlist00, adjlist01, adjlist02, adjlist03]], \
           [[idx00, idx01, idx02, idx03],[idx00, idx01, idx02, idx03]], \
           adjM, type_mask, \
           [drug2id_dict, target2id_dict, disease2id_dict], \
           train_val_test_drug_drug_samples, train_val_test_drug_drug_labels, cellline_expression
#是DNN用的
def load_DNN_DDI_data_te2(prefix='D:/daima/Muthene-main/echino_dataset/fold1/'): # your folder to store the required original data
    print('the path of source file is :', prefix)

    in_file = open(prefix + 'drug2absid_dict.pickle', 'rb')
    drug2id_dict = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + 'target2absid_dict.pickle', 'rb')
    target2id_dict = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + 'disease2absid_dict.pickle', 'rb')
    disease2id_dict = pickle.load(in_file)
    in_file.close()
##    in_file = open(prefix + 'se_symbol2id_dict.pickle', 'rb') # adverse effect name to ids
##    se_symbol2id_dict = pickle.load(in_file)
##    in_file.close()

    train_val_test_drug_drug_samples = np.load(prefix + 'train_val_test_drug_drug_samples.npz')
    train_val_test_drug_drug_labels = np.load(prefix + 'train_val_test_drug_drug_labels.npz')

    if os.path.exists(prefix + 'expression_reduced_normalized.npy'):
        disease_expression = np.load(prefix + 'expression_reduced_normalized.npy', allow_pickle=True)
    else:
        disease_expression = np.zeros((len(disease2id_dict), 1024))
        print('An empty cell line expression data is generated.')

##    return [drug2id_dict,target2id_dict,cellline2id_dict,se_symbol2id_dict,atomnum2id_dict], \
##           train_val_test_drug_drug_samples, train_val_test_drug_drug_labels, all_drug_morgan, cellline_expression
    return [drug2id_dict, target2id_dict, disease2id_dict], \
        train_val_test_drug_drug_samples, train_val_test_drug_drug_labels, disease_expression

# extra data for HNE-GIN (stored in './data/data4training_model/')
## class DrugStrucDataset(DGLDataset):#for HNE-GIN
##     def __init__(self):
##         super().__init__(name='drugstruc')
##
##     def process(self):
##         # the path for reading drug molecular graphs
##         prefix='D:/daima/Muthene-main/data/data4training_model/'
##        # prefix='./data/data4training_model/'
##         print('the path of drug structure file is :', prefix)
##         edges = pd.read_csv(prefix + 'drug_graph_edges.csv')
##         properties = pd.read_csv(prefix + 'drug_graph_properties.csv')
##         nodes = pd.read_csv(prefix + 'drug_graph_nodes.csv')
##         self.graphs = []
##         self.labels = []
##
##         # Create a graph for each graph ID from the edges table.
##         # First process the properties table into two dictionaries with graph IDs as keys.
##         # The label and number of nodes are values.
##         label_dict = {}
##         num_nodes_dict = {}
##             label_dict[row['graph_id']] = row['label']
##             num_nodes_dict[row['graph_id']] = row['num_nodes']
##
##         # For the edges, first group the table by graph IDs.
##         edges_group = edges.groupby('graph_id')
##         nodes_group = nodes.groupby('graph_id')
##
##         # For each graph ID...
##         for graph_id in edges_group.groups:
##             # Find the edges as well as the number of nodes and its label.
##             edges_of_id = edges_group.get_group(graph_id)
##             nodes_of_id = nodes_group.get_group(graph_id)
##             src = edges_of_id['src'].to_numpy()#使用源和目标原子节点创建dgl图
##             dst = edges_of_id['dst'].to_numpy()
##             num_nodes = num_nodes_dict[graph_id]
##             label = label_dict[graph_id]
##             atom_features = torch.from_numpy(nodes_of_id['atom_num'].to_numpy())
##             atom_features = atom_features.type(torch.float32)
##
##            # Create a graph and add it to the list of graphs and labels.
##             g = dgl.graph((src, dst), num_nodes=num_nodes)
##             g.ndata['atom_num'] = atom_features
##             self.graphs.append(g)
##             self.labels.append(label)
##
##         # Convert the label list to tensor for saving.
##         self.labels = torch.LongTensor(self.labels)
##
##     def __getitem__(self, i):
##         return self.graphs[i], self.labels[i]
##
##     def __len__(self):
##         return len(self.graphs)


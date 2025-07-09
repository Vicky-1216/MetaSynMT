import torch.nn as nn
import torch.nn.functional as F
import torch
#from dgl.nn import GINConv, GraphConv
from dgl.nn.pytorch import GINConv, GraphConv ##
import dgl
import numpy as np

# for generating drug structural embedding
class GIN4drug_struc(nn.Module):#gin变体

    def __init__(self, in_feats, h_feats):#in_feats输入维度 即原子类型总数，h_feats输出维度 即每个原子的嵌入向量大小
        super(GIN4drug_struc, self).__init__()
        # in_feats: total number of atom types原子类型的总数
        self.embedding = nn.Embedding(in_feats, h_feats)
        self.lin1 = torch.nn.Linear(h_feats, h_feats)
        self.lin2 = torch.nn.Linear(h_feats, h_feats)
        self.conv1 = GINConv(self.lin1, 'sum')
        self.conv2 = GINConv(self.lin2, 'sum')

    def forward(self, g, in_feat):
        # indices for retrieving embeddings
        h = self.embedding(in_feat)
        h = self.conv1(g, h)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=64, num_features_xd=680, #num_features_xt=731,
                 n_filters=32, embed_dim=128, output_dim=64, dropout=0.2):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output

        self.drug1_conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.drug1_conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*10*2, 512)
        self.drug1_fc_g2 = torch.nn.Linear(512, 128)
        self.drug1_fc_g3 = torch.nn.Linear(128, output_dim)

        # self.drug2_conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        # self.drug2_conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        # self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 10 * 2, 512)
        # self.drug2_fc_g2 = torch.nn.Linear(512, 128)
        # self.drug2_fc_g3 = torch.nn.Linear(128, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


        # DL cell featrues
        # self.reduction = nn.Sequential(
        #     nn.Linear(num_features_xt, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, output_dim)
        # )

        # combined layers
        self.fc1 = nn.Linear(output_dim * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        # x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
        # deal drug1
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x1 = torch.cat([gmp(x1, batch1), gap(x1, batch1)], dim=1)
        x1 = self.relu(self.drug1_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.drug1_fc_g2(x1)
        x1 = self.drug1_fc_g3(x1)
        return x1



class side_effect_predictor(nn.Module):

    def __init__(self, in_feats, h_feats, dropout_rate=0.0):
        # in_feats:输入 dimension of drug embedding (from ECFP6 + from DTI network)
        # h_feats: 输出number of side effects
        super(side_effect_predictor, self).__init__()
        self.lin1 = torch.nn.Linear(in_feats * 2, h_feats)#*2 即之和
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = lambda x: x

        # *** could try extra initialization for all linear layers here ***

    def forward(self, drug_embedding1, drug_embedding2):
        input = torch.cat([drug_embedding1, drug_embedding2], axis=1)
        se_output = self.lin1(self.dropout(input))
        return se_output




class therapeutic_effect_predictor(nn.Module):
    # disese_feats: only for generating the cell line look-up table under the case whether_disese=[False, False]. emd_feats: disese defined dimension
    # in_feats: total embedding dimension for one drug (metapath embedding)
    # layer_list: hidden unit number for each layer (except for input feature number)
    # control whether to concatenate adverse effect outputs
    def __init__(self, disease_gene, similar, disease_feats, in_feats, emd_feats, layer_list, output_concat=False, dropout=0.0, input_dropout=0.0, whether_disease=[True, True]):
        #print('TE predictor hyper-paramters:', disease_feats, in_feats, emd_feats, layer_list, output_concat, dropout, input_dropout, whether_disease)    ## output_concat, concat_feats, dropout
        super(therapeutic_effect_predictor, self).__init__() #cellline_feats细胞系特征的维度 in_feats每个药物的总嵌入维度 emd_feats：细胞系嵌入的维度 output_concat：是否连接不良反应 whether_CCLE：是否使用细胞系表达数据和是否进行维度缩减的标志位
        self.whether_disease = whether_disease
        self.emd_feats = emd_feats

        # whether_disease[0]: True: use true cell line expression data. False: use one-hot encoding instead
        # whether_disease[1]:
        # True: directly use cell line expression data without dimension reduction ('--hidden-dim-aux' fails)
        # False: use cell line expression data with dimension reduction, the reduced dimension is determined by '--hidden-dim-aux'

        # need to explain how to leverage cell line related information in detail
        if self.whether_disease[0] == True:#使用真实的细胞系表达数据
            self.disease_gene = disease_gene
            self.similar = similar

            self.emd_feats = self.disease_gene.size(1) #列数
            if self.whether_disease[1] == False:
                self.emd_feats = emd_feats
                self.similar_transform = nn.Sequential(
                    nn.Linear(self.similar.shape[1], self.emd_feats, bias=True),  # 输入层(25 64到64 64)
                    nn.ReLU(),  # 激活函数
                    nn.Linear(self.emd_feats, self.emd_feats)  # 输出层
                )
                self.disease = nn.Sequential(
                    nn.Linear(self.disease_gene.size(1), self.emd_feats * 2, bias=True),  # 输入层
                    nn.ReLU(),  # 激活函数
                    nn.Linear(self.emd_feats * 2, self.emd_feats)  # 输出层
                )
                # similar_feat = self.similar_transform(self.similar)
                # disese_feat = self.disese(self.disese_expression)

        else:
            ##self.cellline_transform = nn.Embedding(cellline_feats, self.emd_feats) # cell line number * cell line dimension
            self.disese_transform = nn.Sequential(
                nn.Linear(self.disease_expression.size(1), self.emd_feats, bias=True),  # 输入层
                nn.ReLU(),  # 激活函数
                nn.Linear(self.emd_feats, self.emd_feats)  # 输出层
            )

        # drug-drug-cell line pair encoding:
        self.linears = nn.ModuleList()
        for i in range(len(layer_list)):
            if i == 0: # the first layer
                if output_concat == True:#需要拼接se
                    #print('Neurons in first layer of TE predictor:', in_feats * 2 + self.emd_feats * 2, 'disease dimension:', self.emd_feats * 2) ## self.emd_feats + concat_feats
                    self.linears.append(torch.nn.Linear(in_feats * 2 + self.emd_feats * 2, layer_list[i]))   ## self.emd_feats + concat_feats
                else:
                    #print('Neurons in first layer of TE predictor:', in_feats * 2 + self.emd_feats * 2, 'disease dimension:', self.emd_feats * 2)
                    self.linears.append(torch.nn.Linear(in_feats * 2 + self.emd_feats * 2, layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(input_dropout))
            elif i == len(layer_list) - 1: # the last layer
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
            else: # the intermediate layers
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(dropout))

        # *** could try extra initialization for all linear layers here ***
        # for fc in self.linears:
        #     if isinstance(fc, nn.Linear):
        #         nn.init.xavier_normal_(fc.weight, gain=1.414)
        # if (self.whether_disese[0] == True) and (self.whether_disese[1] == False):
        #     nn.init.xavier_normal_(self.disese_transform.weight, gain=1.414)

    def forward(self, drug_embedding1, drug_embedding2, disease_idx, se_output=None):#se_output来自se任务
        # disease encoding
        if (self.whether_disease[0] == True) and (self.whether_disease[1] == True):
            # use cell line information directly
            disease_embedding1 = self.disease_gene[disease_idx]
            disease_embedding2 = self.similar[disease_idx]

        elif (self.whether_disease[0] == True) and (self.whether_disease[1] == False):
            disease_embedding_encoding1 = self.disease(self.disease_gene)
            disease_embedding_encoding2 = self.similar_transform(self.similar)
            disease_embedding1 = disease_embedding_encoding1[disease_idx]
            disease_embedding2 = disease_embedding_encoding2[disease_idx]
            disease_embedding = torch.cat((disease_embedding1, disease_embedding2), axis=1)
        else:
            disease_embedding = self.disease_transform(disease_idx)

        # feature concatenation
        if se_output != None:
            input = torch.cat((drug_embedding1, drug_embedding2, disease_embedding, se_output), axis=1)
        else:
            input = torch.cat((drug_embedding1, drug_embedding2, disease_embedding), axis=1)

        # drug-drug-disease line pair encoding
        for layer in self.linears:
            input = layer(input)
        return input

class side_effect_predictor2(nn.Module):
    # in_feats: total embedding dimension for one drug (metapath embedding)
    # layer_list: hidden unit number for each layer (except for input feature number)
    # control whether to concatenate adverse effect outputs
    def __init__(self, in_feats, layer_list, dropout=0.0, input_dropout=0.0):# output_concat=False, emd_feats,
        #print('SE predictor hyper-paramters:', in_feats,layer_list, dropout, input_dropout)    ## output_concat, concat_feats, dropout
        super(side_effect_predictor2, self).__init__() #cellline_feats细胞系特征的维度 in_feats每个药物的总嵌入维度 emd_feats：细胞系嵌入的维度 output_concat：是否连接不良反应 whether_CCLE：是否使用细胞系表达数据和是否进行维度缩减的标志位

        self.linears = nn.ModuleList()
        for i in range(len(layer_list)):
            if i == 0: # the first layer
                ## if output_concat == True:#需要拼接se
                ##     print('Neurons in first layer of SE predictor:', in_feats * 2 ) ## self.emd_feats + concat_feats
                ##     self.linears.append(torch.nn.Linear(in_feats * 2 , layer_list[i]))   ## self.emd_feats + concat_feats
                ## else:
                ##     print('Neurons in first layer of SE predictor:', in_feats * 2 )
                ##     self.linears.append(torch.nn.Linear(in_feats * 2 , layer_list[i]))
                #print('Neurons in first layer of SE predictor:', in_feats * 2)
                self.linears.append(torch.nn.Linear(in_feats * 2, layer_list[i]))##
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(input_dropout))
            elif i == len(layer_list) - 1: # the last layer
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
            else: # the intermediate layers
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(dropout))


    def forward(self, drug_embedding1, drug_embedding2):#, se_output=None):#se_output来自se任务

        # feature concatenation
        ##if se_output != None:
            ##input = torch.cat((drug_embedding1, drug_embedding2, se_output), axis=1)
        ##else:
        input = torch.cat((drug_embedding1, drug_embedding2), axis=1)

        # drug-drug-cell line pair encoding
        for layer in self.linears:
            input = layer(input)
        return input
class side_effect_predictor(nn.Module):#前面的

    def __init__(self, in_feats, layer_list, dropout=0.0, input_dropout=0.0):
        super(side_effect_predictor, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layer_list)):
            if i == 0:
                self.linears.append(torch.nn.Linear(in_feats * 2, layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(input_dropout))
            elif i == len(layer_list) - 1:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
            else:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(dropout))

    def forward(self, drug_embedding1, drug_embedding2):
        input = torch.cat([drug_embedding1, drug_embedding2], axis=1)
        for layer in self.linears:
            input = layer(input)
        return input


# for automatically balancing weights of two tasks
class AutomaticWeightedLoss(nn.Module):#自动加权的多任务损失函数
    """automatically weighted multi-task loss
    Params：
        num: int,the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):#两个损失
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)#可训练参数
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)#群众参数自动调整 损失大的权重小
        return loss_sum

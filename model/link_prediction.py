import torch
import torch.nn as nn
import numpy as np
from model.base_model import ntype_specific


# for link prediction task预测新的药物药物相互作用
class layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='rnn',#指定rnn的类型
                 attn_drop=0.5,
                 attn_switch=False,
                 rnn_concat=False):#影响着多头注意力输出的特征维度 形成更大的特征向量或每个头的输出保持独立
        super(layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # drug/target specific layers获得
        self.drug_layer = ntype_specific(num_metapaths_list[0],#获得metapath-specific 节点特征表示 h，以及用于注意力机制的权重 beta
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   attn_drop,
                                                   use_minibatch=True,
                                                   attn_switch=attn_switch,
                                                   rnn_concat=rnn_concat)

        # note that the actual input dimension should consider the number of heads as multiple head outputs are concatenated together实际的输入维度应该考虑多个头部输出连接在一起时的头部数量
        if (rnn_concat == True):#否应该在 RNN 的输出中使用拼接 双向的就得拼接
            self.fc_drug = nn.Linear(in_dim * num_heads * 2, out_dim, bias=True)
            # self.fc_target = nn.Linear(in_dim * num_heads * 2, out_dim, bias=True)
        else:
            self.fc_drug = nn.Linear(in_dim * num_heads, out_dim, bias=True)
            # self.fc_target = nn.Linear(in_dim * num_heads, out_dim, bias=True)

        nn.init.xavier_normal_(self.fc_drug.weight, gain=1.414)#初始化全连接层的权重
        # nn.init.xavier_normal_(self.fc_target.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs
        # drug/target specific layers  #边元路径索引列表和靶标索引列表。
        h_drug1, atten_drug1 = self.drug_layer( #药物的输出特征和注意力权重
        #h_drug1=self.drug_layer(  # 药物的输出特征和注意力权重
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0]))
        h_drug2, atten_drug2 = self.drug_layer(
        #h_drug2=self.drug_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1]))

        logits_drug1 = self.fc_drug(h_drug1)#预测的逻辑结果
        logits_drug2 = self.fc_drug(h_drug2)
        return [logits_drug1, logits_drug2], [h_drug1, h_drug2], [atten_drug1, atten_drug2]
#包括一个用于处理药物特征的神经网络层和一个全连接层

class link_prediction(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='rnn',
                 dropout_rate=0.5,
                 attn_switch=False,
                 rnn_concat=False,
                 args=None):
        super(link_prediction, self).__init__()
        self.hidden_dim = hidden_dim
        self.args = args

        # node type specific transformation特定于节点类型的转换 两个类型 drug和target 并且两个输入维度不一样但输出维度都是64
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])#创建了一个包含多个线性层的列表，并将其封装在 ModuleList
        # feature dropout after transformation                                                                    #每个 feats_dim 创建一个输入维度为 feats_dim、输出维度为 hidden_dim 的线性层
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.layer = layer(num_metapaths_list,#用于执行药物相互作用预测的一层操作
                                     hidden_dim,
                                     out_dim,
                                     num_heads,
                                     attn_vec_dim,
                                     rnn_type,
                                     attn_drop=dropout_rate,
                                     attn_switch=attn_switch,
                                     rnn_concat=rnn_concat)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # node type specific transformation 将节点特征进行类型特定的转换
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)#创建一个形状为 (样本数, hidden_dim) 的全零张量，
        for i, fc in enumerate(self.fc_list):#遍历节点类型对应的线性层列表
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        # create a matrix storing all node features of the dataset
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers
        [logits_drug1, logits_drug2], [h_drug1, h_drug2], [atten_drug1, atten_drug2] = self.layer((g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))
        #[logits_drug1, logits_drug2], [h_drug1, h_drug2]= self.layer(
        # Print attention scores for different metapaths
        # for i, atten in enumerate([atten_drug1, atten_drug2]):
        #     print(f"Attention scores for metapath {i + 1}:")
        #     print(atten)
        #
        # # Print drug names (assuming drug names are stored in features_list)
        # for i, drug_features in enumerate(features_list):
        #     if i == 0:
        #         print("Drug names:")
        #     drug_indices = np.where(type_mask == i)[0]
        #     drugs = drug_features[drug_indices]
        #     print(f"Drug type {i}: {drugs}")
        # attention_scores = [atten_drug1, atten_drug2]
        # for i, score in enumerate(attention_scores):
        #     # 假设每个得分是一个张量，我们打印其值
        #     print(f"Attention scores for meta-path {i}: {score}")

        return [logits_drug1, logits_drug2], [h_drug1, h_drug2], [atten_drug1, atten_drug2]



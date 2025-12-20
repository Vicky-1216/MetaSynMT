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


class side_effect_predictor(nn.Module):

    def __init__(self, in_feats, h_feats, dropout_rate=0.0):
        # in_feats:输入 dimension of drug embedding
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


class DNN_predictor(nn.Module):
    def __init__(self, cellline_expression, in_feats, cellline_feats, emd_feats, layer_list, dropout, input_dropout, whether_CCLE=[True, True]):
        print('DNN predictor hyper-paramters:', in_feats, cellline_feats, emd_feats, layer_list, dropout, input_dropout, whether_CCLE)
        super(DNN_predictor, self).__init__()     #emd_feats：cell line defined dimension嵌入特征的维度 layer_list：DNN 中每个隐藏层的神经元数量列表
        self.whether_CCLE = whether_CCLE
        self.emd_feats = emd_feats
        if self.whether_CCLE[0] == True:
            self.cellline_expression = cellline_expression
            self.emd_feats = self.cellline_expression.size(1)
            if self.whether_CCLE[1] == False:
                self.emd_feats = emd_feats
                self.cellline_transform = nn.Linear(self.cellline_expression.size(1), self.emd_feats, bias=True) # for feature reduction
        else:
            self.cellline_transform = nn.Embedding(cellline_feats, self.emd_feats) # cell line number --> len(cellline2id_dict) * cell line dimension

        self.linears = nn.ModuleList()
        for i in range(len(layer_list)):#遍历神经网络的层列表 多层线性变换层的构建。
            if i == 0:
                print('Neurons in first layer of DNN predictor:', in_feats + self.emd_feats, 'disease dimension:', self.emd_feats)
                self.linears.append(torch.nn.Linear(in_feats + self.emd_feats, layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(input_dropout))
            elif i == len(layer_list) - 1:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
            else:
                self.linears.append(torch.nn.Linear(layer_list[i - 1], layer_list[i]))
                self.linears.append(torch.nn.ReLU())
                self.linears.append(nn.Dropout(dropout))

    def forward(self, drug_embedding1, drug_embedding2, cellline_idx, se_labels_batch=None): #se_labels_batch是se任务得到的
        if (self.whether_CCLE[0] == True) and (self.whether_CCLE[1] == True):
            # use cell line information directly
            cellline_embedding = self.cellline_expression[cellline_idx]
        elif (self.whether_CCLE[0] == True) and (self.whether_CCLE[1] == False):
            cellline_embedding_encoding = self.cellline_transform(self.cellline_expression)
            cellline_embedding = cellline_embedding_encoding[cellline_idx]
        else:
            cellline_embedding = self.cellline_transform(cellline_idx)#形状为 (batch_size, num_celllines) 的张量

        if se_labels_batch != None: # in the case for fusing adverse effect labels
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding, se_labels_batch), axis=1)#拼接操作
        else:
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding), axis=1)

        for layer in self.linears:
            input = layer(input)
        return input


class therapeutic_effect_DNN_predictor(nn.Module):

##  def __init__(self, cellline_expression, cellline_feats, in_feats, emd_feats, layer_list, output_concat=False, concat_feats=0, dropout=0.0, input_dropout=0.0, whether_CCLE=[True, True]):
    def __init__(self, cellline_expression, similar, cellline_feats, in_feats, emd_feats, layer_list, output_concat=False, dropout=0.0, input_dropout=0.0, whether_CCLE=[True, True]):
        print('TE predictor hyper-paramters:', cellline_feats, in_feats, emd_feats, layer_list, output_concat, dropout, input_dropout, whether_CCLE)    ## output_concat, concat_feats, dropout
        super(therapeutic_effect_DNN_predictor, self).__init__() #cellline_feats细胞系特征的维度 in_feats每个药物的总嵌入维度 emd_feats：细胞系嵌入的维度 output_concat：是否连接不良反应 whether_CCLE：是否使用细胞系表达数据和是否进行维度缩减的标志位
        self.whether_CCLE = whether_CCLE
        self.emd_feats = emd_feats


        # need to explain how to leverage cell line related information in detail
        if self.whether_CCLE[0] == True:
            self.cellline_expression = cellline_expression
            self.similar = similar

            self.emd_feats = self.cellline_expression.size(1) #列数
            if self.whether_CCLE[1] == False:
                self.emd_feats = emd_feats
                ##self.cellline_transform = nn.Linear(self.cellline_expression.size(1), self.emd_feats, bias=True) # for feature reduction
                self.similar_transform = nn.Sequential(
                    nn.Linear(self.similar.shape[1], self.emd_feats, bias=True),  # 输入层(25 64到64 64)
                    nn.ReLU(),  # 激活函数
                    nn.Linear(self.emd_feats, self.emd_feats)  # 输出层
                )
                self.cellline = nn.Sequential(
                    nn.Linear(self.cellline_expression.size(1), self.emd_feats * 2, bias=True),  # 输入层
                    nn.ReLU(),  # 激活函数
                    nn.Linear(self.emd_feats * 2, self.emd_feats)  # 输出层
                )
                # similar_feat = self.similar_transform(self.similar)
                # celline_feat = self.cellline(self.cellline_expression)

        else:
            ##self.cellline_transform = nn.Embedding(cellline_feats, self.emd_feats) # cell line number * cell line dimension
            self.cellline_transform = nn.Sequential(
                nn.Linear(self.cellline_expression.size(1), self.emd_feats, bias=True),  # 输入层
                nn.ReLU(),  # 激活函数
                nn.Linear(self.emd_feats, self.emd_feats)  # 输出层
            )

        # drug-drug-cell line pair encoding:
        self.linears = nn.ModuleList()
        for i in range(len(layer_list)):
            if i == 0: # the first layer
                if output_concat == True:#需要拼接se
                    print('Neurons in first layer of TE predictor:', in_feats * 2 + self.emd_feats * 2, 'disease dimension:', self.emd_feats * 2) ## self.emd_feats + concat_feats
                    self.linears.append(torch.nn.Linear(in_feats * 2 + self.emd_feats * 2, layer_list[i]))   ## self.emd_feats + concat_feats
                else:
                    print('Neurons in first layer of TE predictor:', in_feats * 2 + self.emd_feats * 2, 'disease dimension:', self.emd_feats * 2)
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
        # if (self.whether_CCLE[0] == True) and (self.whether_CCLE[1] == False):
        #     nn.init.xavier_normal_(self.cellline_transform.weight, gain=1.414)

    def forward(self, drug_embedding1, drug_embedding2, cellline_idx, se_output=None):#se_output来自se任务
        # cell line encoding
        if (self.whether_CCLE[0] == True) and (self.whether_CCLE[1] == True):
            # use cell line information directly
            cellline_embedding1 = self.cellline_expression[cellline_idx]
            cellline_embedding2 = self.similar[cellline_idx]

        elif (self.whether_CCLE[0] == True) and (self.whether_CCLE[1] == False):
            cellline_embedding_encoding1 = self.cellline(self.cellline_expression)
            cellline_embedding_encoding2 = self.similar_transform(self.similar)
            cellline_embedding1 = cellline_embedding_encoding1[cellline_idx]
            cellline_embedding2 = cellline_embedding_encoding2[cellline_idx]
            cellline_embedding = torch.cat((cellline_embedding1, cellline_embedding2), axis=1)
        else:
            cellline_embedding = self.cellline_transform(cellline_idx)

        # feature concatenation
        if se_output != None:
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding, se_output), axis=1)
        else:
            input = torch.cat((drug_embedding1, drug_embedding2, cellline_embedding), axis=1)

        # drug-drug-cell line pair encoding
        for layer in self.linears:
            input = layer(input)
        return input

class side_effect_DNN_predictor2(nn.Module):
##  def __init__(self, cellline_expression, cellline_feats, in_feats, emd_feats, layer_list, output_concat=False, concat_feats=0, dropout=0.0, input_dropout=0.0, whether_CCLE=[True, True]):
    def __init__(self, in_feats, layer_list, dropout=0.0, input_dropout=0.0):# output_concat=False, emd_feats,
        print('SE predictor hyper-paramters:', in_feats,layer_list, dropout, input_dropout)    ## output_concat, concat_feats, dropout
        super(side_effect_DNN_predictor2, self).__init__() #cellline_feats细胞系特征的维度 in_feats每个药物的总嵌入维度 emd_feats：细胞系嵌入的维度 output_concat：是否连接不良反应 whether_CCLE：是否使用细胞系表达数据和是否进行维度缩减的标志位
        ##self.whether_CCLE = whether_CCLE
        ##self.emd_feats = emd_feats

        # drug-drug-cell line pair encoding:
        self.linears = nn.ModuleList()
        for i in range(len(layer_list)):
            if i == 0: # the first layer
                ## if output_concat == True:#需要拼接se
                ##     print('Neurons in first layer of SE predictor:', in_feats * 2 ) ## self.emd_feats + concat_feats
                ##     self.linears.append(torch.nn.Linear(in_feats * 2 , layer_list[i]))   ## self.emd_feats + concat_feats
                ## else:
                ##     print('Neurons in first layer of SE predictor:', in_feats * 2 )
                ##     self.linears.append(torch.nn.Linear(in_feats * 2 , layer_list[i]))
                print('Neurons in first layer of SE predictor:', in_feats * 2)
                self.linears.append(torch.nn.Linear(in_feats * 2, layer_list[i]))##
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
        # if (self.whether_CCLE[0] == True) and (self.whether_CCLE[1] == False):
        #     nn.init.xavier_normal_(self.cellline_transform.weight, gain=1.414)

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
class side_effect_DNN_predictor(nn.Module):

    def __init__(self, in_feats, layer_list, dropout=0.0, input_dropout=0.0):
        super(side_effect_DNN_predictor, self).__init__()
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

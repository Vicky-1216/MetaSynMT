import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
import copy
from model.Trans_encoder import MultiHeadedAttention, PositionwiseFeedForward, Encoder, EncoderLayer


# class BiCNN(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, hidden_dim: int, kernel_size = 3, stride = 1, bias: bool = True,bidirectional: bool = False):
#         super(BiCNN, self).__init__()
#         # 第一个卷积层（正向）
#         #self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
#         # 第二个卷积层（反向）
#         self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
#         self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
#         #self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
#         # 隐藏层
#         self.hidden_layer = nn.Linear(out_channels * 2, hidden_dim)
#
#     def forward(self, x):
#         # 正向卷积
#         out1 = self.conv1(x)
#         # 反向卷积（将输入翻转）
#         out2 = self.conv2(torch.flip(x, [2]))
#         # 将正向和反向的输出拼接在一起
#         out = torch.cat((out1, out2), dim=1)
#         out = out.view(out.size(0), -1)
#         out = self.hidden_layer(out)
#         return out
# class YourCNN(nn.Module):
#     def __init__(self, ...):
#         super(YourCNN, self).__init__()
#         self.conv = nn.Conv1d(...)
#         self.hidden_layer = nn.Linear(...)
#         # 其他初始化操作
#
#     def forward(self, x):
#         # 卷积操作
#         conv_output = self.conv(x)
#
#         # 添加类似于隐藏状态的操作
#         hidden = self.hidden_layer(conv_output.mean(dim=1))
#
#         return conv_output, hidden



class HNEMA_metapath_specific(nn.Module):#specific 用于处理元路径特定的节点嵌入学习任务
    def __init__(self,
                 out_dim,
                 #hidden_dim,
                 num_heads,#
                 rnn_type='rnn',
                 attn_drop=0.5,#
                 ##hidden_dim
                 alpha=0.01,#LeakyReLU 函数的负斜率参数
                 use_minibatch=False,
                 attn_switch=False,#
                 rnn_concat=False):#。
        super(HNEMA_metapath_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads#
        self.rnn_type = rnn_type
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch#
        self.rnn_concat = rnn_concat
        #self.hidden_dim = hidden_dim
        out_dim = int(out_dim)#
        num_heads = int(num_heads)#
        # rnn-like metapath instance aggregator
        # consider multiple attention heads
        if rnn_type == 'bi-lstm':
            print('current rnn type is:', rnn_type)
            self.rnn = nn.LSTM(out_dim,  out_dim // 2, bidirectional=True)
            #self.rnn = nn.LSTM(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'lstm':
            print('current rnn type is:', rnn_type)
            self.rnn = nn.LSTM(out_dim, out_dim)
            #self.rnn = nn.LSTM(out_dim, num_heads * out_dim)
        elif rnn_type == 'bi-gru':
            print('current rnn type is:', rnn_type)
            self.rnn = nn.GRU(out_dim,  out_dim // 2, bidirectional=True)
            #self.rnn = nn.GRU(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'cnn':
            print('current rnn type is:', rnn_type)
            self.conv = nn.Conv1d(out_dim,  out_dim , kernel_size = 3, stride = 1)#, bidirectional=True)
            self.trans_out = nn.Linear(out_dim,  out_dim, bias=False)
            # self.conv = nn.Conv1d(out_dim, num_heads * out_dim, kernel_size=3, stride=1)  # , bidirectional=True)
            # self.trans_out = nn.Linear(out_dim, num_heads * out_dim, bias=False)
            #self.hidden_layer = nn.Linear(num_heads * out_dim , hidden_dim)
        elif rnn_type == 'rnn':
            print('current rnn type is:', rnn_type)
            #self.rnn = nn.RNN(out_dim,  out_dim)#
            self.rnn = nn.RNN(out_dim, num_heads * out_dim)
        elif rnn_type == 'gru':
            print('current rnn type is:', rnn_type)
            self.rnn = nn.GRU(out_dim, out_dim)
            #self.rnn = nn.GRU(out_dim, num_heads * out_dim)
        elif rnn_type == 'mean':
            self.trans_out1 = nn.Linear(out_dim, out_dim, bias=False)
            #self.trans_out1 = nn.Linear(out_dim, num_heads * out_dim, bias=False)
            nn.init.xavier_normal_(self.trans_out1.weight, gain=1.414)
        elif rnn_type == 'transformer':
            c = copy.deepcopy
            # attn = MultiHeadedAttention(num_heads, out_dim)
            attn = MultiHeadedAttention(1, out_dim)

            # the size of fc equals to the output of multi-attention layer
            # num_heads: 64, out_dim:8, fc_hidden_state, dropout

            # ff = PositionwiseFeedForward(num_heads * out_dim, 512, 0.1)
            ff = PositionwiseFeedForward(out_dim, 512, 0.1)

            # self.skip_proj = nn.Linear(out_dim, num_heads * out_dim, bias=False)
            # the second parameter represents the number of encoder block of transformer extractor
            self.rnn = Encoder(EncoderLayer(out_dim, c(attn), c(ff), 0.1), 6)

            #self.trans_out1 = nn.Linear(out_dim,  out_dim, bias=False)#
            #self.trans_out2 = nn.Linear(out_dim,  out_dim, bias=False)#
            self.trans_out1 = nn.Linear(out_dim, num_heads * out_dim, bias=False)
            self.trans_out2 = nn.Linear(out_dim, num_heads * out_dim, bias=False)

            for p in self.rnn.parameters():
                if (p.dim() > 1):
                    nn.init.xavier_uniform_(p)

            # nn.init.xavier_normal_(self.skip_proj.weight, gain=1.414)
            nn.init.xavier_normal_(self.trans_out1.weight, gain=1.414)
            nn.init.xavier_normal_(self.trans_out2.weight, gain=1.414)

        # elif rnn_type == 'gat':
        #     self.trans_out1 = nn.Linear(out_dim, num_heads * out_dim, bias=False)
        #     nn.init.xavier_normal_(self.trans_out1.weight, gain=1.414)

        # node-level attention节点级注意
        # attention considers the center node embedding or not
        if self.attn_switch:# #为True，则表示要使用两个不同的注意力参数
            # self.attn1 = nn.Linear(out_dim, num_heads, bias=False)
            self.attn1 = nn.Linear(num_heads * out_dim, num_heads, bias=False)# # Linear(in_features=512, out_features=8)
            self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))#
        else:
            self.attn = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))#
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:#
            self.attn_drop = nn.Dropout(attn_drop)#
        else:#
            self.attn_drop = lambda x: x # #否则被设成一个恒等的不修改

        # weight initialization
        if self.attn_switch:#
            nn.init.xavier_normal_(self.attn1.weight, gain=1.414)#
            nn.init.xavier_normal_(self.attn2.data, gain=1.414)#
        else:#
            nn.init.xavier_normal_(self.attn.data, gain=1.414)#

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a')) #从图的边特征中提取出注意力分数
        g.edata['a_drop'] = self.attn_drop(attention)

    def message_passing(self, edges): #定义了消息传递规则，用于计算节点聚合特征
        #ft = edges.data['eft'] #* edges.data['a_drop']
        ft = edges.data['eft'] * edges.data['a_drop']
        return {'ft': ft}

    def message_passing_bi_lstm(self, edges): #定义了用于双向的消息传递规则
        avg = edges.data['eft']
        return {'avg': avg}

    def forward(self, inputs):
        if self.use_minibatch:
            g, features, type_mask, edge_metapath_indices, target_idx = inputs #边元路径索引
        else:
            g, features, type_mask, edge_metapath_indices = inputs

        # Embedding layer
        # use torch.nn.functional.embedding or torch.embedding here
        # do not use torch.nn.embedding
        # edata: E x Seq x out_dim
        edata = F.embedding(edge_metapath_indices, features)

        # apply rnn to metapath-based feature sequence
        if self.rnn_type == 'bi-lstm':
            # the size of output is  [sequence length, batch size, num_heads (e.g.,8) * out_dim (e.g.,64) ]
            output, (hidden, _) = self.rnn(edata.permute(1, 0, 2))#permute交换维度

            # if (self.attn_switch == True):
            #     # source and target node embeddings contain the separate word embedding through the bidirectional learning
            #     # hidden contains comprehensive information extracted from the whole sequence
            #     # source_node_embed = output[0]
            #     target_node_embed = output[-1]
            #
            #     # in this case, torch.split equals to tensor.reshape
            #     target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(#reshape
            #         0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
            #     hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
            #         0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
            #
            # else:
            #     hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
            #         0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
                0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'bi-gru':
            output, hidden = self.rnn(edata.permute(1, 0, 2))
            if (self.attn_switch == True):
                # source and target node embeddings contain the separate word embedding through the bidirectional learning
                # hidden contains comprehensive information extracted from the whole sequence
                # source_node_embed = output[0]
                target_node_embed = output[-1]
                # print(output.size())
                # torch.Size([3, 62, 512])

                # in this case, torch.split equals to tensor.reshape
                target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
                hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

            else:
                hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'cnn':
            ##output, hidden = self.rnn(edata.permute(1, 0, 2))
            #output, hidden = self.conv(edata.permute(1, 0, 2))
            output = self.conv(edata.permute(1, 0, 2))
            hidden = self.leaky_relu(self.trans_out(output))
            #hidden = self.hidden_layer(output.mean(dim=1))
            if (self.attn_switch == True):
                # source and target node embeddings contain the separate word embedding through the bidirectional learning
                # hidden contains comprehensive information extracted from the whole sequence
                # source_node_embed = output[0]
                target_node_embed = output[-1]
                # print(output.size())
                # torch.Size([3, 62, 512])

                # in this case, torch.split equals to tensor.reshape
                target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
                hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1,self.num_heads * self.out_dim).unsqueeze(dim=0)
                ##hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
                ##    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

            else:
                hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
                ##hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(
                ##    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'lstm':#长短时记忆网络
            output, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            if (self.attn_switch == True):
                target_node_embed = output[-1]
                target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'rnn':
            output, hidden = self.rnn(edata.permute(1, 0, 2))

            if (self.attn_switch == True):#
                target_node_embed = output[-1]#
                # target_node_embed = target_node_embed.reshape(-1, self.out_dim).permute( #
                #     0, 2, 1).reshape(-1, self.out_dim).unsqueeze(dim=0) #
                target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'gru':
            output, hidden = self.rnn(edata.permute(1, 0, 2))
            if (self.attn_switch == True):
                target_node_embed = output[-1]
                target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

        elif self.rnn_type == 'mean':
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)

            if (self.attn_switch == True):
                edata = self.trans_out1(edata)
                target_node_embed = edata[:, -1, :]
                target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                    0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

        elif self.rnn_type == 'transformer':
            output = self.rnn(edata, None)
            target_node_embed = output[:, -1, :]

            target_node_embed = self.trans_out1(target_node_embed)
            hidden = self.leaky_relu(self.trans_out2(output.mean(dim=1)))

            target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
                0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
            hidden = hidden.reshape(-1, self.out_dim, self.num_heads).permute(
                0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)

        # elif self.rnn_type == 'gat':
        #     edata = self.trans_out1(edata)
        #     source_node_embed = edata[:, 0, :]
        #     target_node_embed = edata[:, -1, :]
        #     source_node_embed = source_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
        #         0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        #     target_node_embed = target_node_embed.reshape(-1, self.out_dim, self.num_heads).permute(
        #         0, 2, 1).reshape(-1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        #     hidden = source_node_embed

        eft = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.out_dim)  # E x num_heads x out_dim #bigru的隐藏状态
        #eft = hidden.permute(1, 0, 2).view(-1, self.out_dim)#
        if self.attn_switch:# #是否用两个
            center_node_feat = target_node_embed.squeeze(dim=0)# #挤压维度为 0 的轴，得到中心节点特征 center_node_feat
            a1 = self.attn1(center_node_feat)# # E x num_heads #注意力系数
        #     # self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, out_dim))), eft=E x num_heads x out_dim
        #     # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
            a2 = (eft * self.attn2).sum(dim=-1)# # E x num_heads #对 eft 进行加权求和，并通过求和操作得到注意力系数 a2
            a = (a1 + a2).unsqueeze(dim=-1)# # E x num_heads x 1 三维
        else:
            a = (eft * self.attn).sum(dim=-1).unsqueeze(dim=-1)# # E x num_heads x 1

        a = self.leaky_relu(a)#
        # switch the device
       # g = g.to(torch.device('cuda:0'))
        g = g.to(torch.device('cpu'))

        #g.edata.update({'eft': eft})#
        g.edata.update({'eft': eft, 'a': a})#
        self.edge_softmax(g)#

        # compute the aggregated node features scaled by the dropped, unnormalized attention values.聚合节点特征
        # Send messages along all the edges of the specified type and update all the nodes of the corresponding destination type.更新相应类型的所有节点
        g.update_all(self.message_passing, fn.sum('ft', 'ft')) #ft是
        ret = g.ndata['ft']  # E x num_heads x out_dim

        if (self.rnn_concat == True):
            g.update_all(self.message_passing_bi_lstm, fn.mean('avg', 'avg'))
            aux = g.ndata['avg']

        if self.use_minibatch:
            if (self.rnn_concat == True):
                return torch.cat([ret[target_idx], aux[target_idx]], dim=-1)
            else:
                return ret[target_idx]
        else:
            return ret


class HNEMA_ctr_ntype_specific(nn.Module):#用于处理多种metapath的模型 #引用HNEMA_metapath_specific
    def __init__(self,
                 num_metapaths,
                 out_dim,
                 #hidden_dim,
                 num_heads,#
                 attn_vec_dim,#
                 rnn_type='rnn',
                 attn_drop=0.5,#
                 use_minibatch=False,
                 attn_switch=False,#
                 rnn_concat=False):#  是否双向的RNN模型（如双向GRU）
        super(HNEMA_ctr_ntype_specific, self).__init__()
        self.out_dim = out_dim
        #self.hidden_dim = hidden_dim
        self.num_heads = num_heads#
        self.use_minibatch = use_minibatch
        self.rnn_concat = rnn_concat

        # metapath-specific layers
        self.metapath_layers = nn.ModuleList() #创建一个ModuleList，用于存储多个metapath-specific的子模型
        for i in range(num_metapaths):#循环创建num_metapaths个HNEMA_metapath_specific的实例，并添加到metapath_layers中。
            self.metapath_layers.append(HNEMA_metapath_specific(out_dim,#每个metapath调用一个
                                                                #hidden_dim,
                                                                num_heads,#
                                                                rnn_type,
                                                                attn_drop=attn_drop,#
                                                                use_minibatch=use_minibatch,
                                                                attn_switch=attn_switch,#
                                                                rnn_concat=rnn_concat))

        # metapath-level attention
        # note that the actual input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        if (self.rnn_concat == True): #全连接层fc1 fc2
            # self.fc1 = nn.Linear(out_dim  * 2, 64, bias=True)#
            # self.fc2 = nn.Linear(64, 1, bias=False)#
            self.fc1 = nn.Linear(out_dim * num_heads * 2, attn_vec_dim, bias=True)
            self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
        else:
            # self.fc1 = nn.Linear(out_dim , 64, bias=True)#
            # self.fc2 = nn.Linear(64, 1, bias=False)#
            self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
            self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)#fc1和fc2的权重进行Xavier正态分布初始化
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, inputs):
        if self.use_minibatch:
            g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs
                            #type_mask标识图中节点的类型  边元路径索引列表 edge_metapath_indices_list
            # metapath-specific layers
            if self.rnn_concat == True: #使用了双向的RNN模型（如双向GRU）
                metapath_outs = [
                    F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,#激活函数
                                                                                                           self.num_heads * self.out_dim * 2))#使用了双向的RNN模型（如双向GRU）
                    for g, edge_metapath_indices, target_idx, metapath_layer in
                    zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
            else:
                metapath_outs = [
                    F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,
                                                                                                           self.num_heads * self.out_dim))#

                    #F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,
                                                                                                          #  self.out_dim))

                for g, edge_metapath_indices, target_idx, metapath_layer in
                    zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]

        else:
            g_list, features, type_mask, edge_metapath_indices_list = inputs

            # metapath-specific layers
            metapath_outs = [F.elu(
                metapath_layer((g, features, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim))
                for g, edge_metapath_indices, metapath_layer in
                zip(g_list, edge_metapath_indices_list, self.metapath_layers)]

        beta = []# #beta列表，用于存储各个metapath的注意力权重。
        # all the metapaths share the same fc1 and fc2
        for metapath_out in metapath_outs:
            fc1 = torch.tanh(self.fc1(metapath_out))
            # calculate the mean value of this metapath
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)#

        beta = torch.cat(beta, dim=0)#
        beta = F.softmax(beta, dim=0)#
        beta = torch.unsqueeze(beta, dim=-1)#
        beta = torch.unsqueeze(beta, dim=-1)#

        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in metapath_outs]

        metapath_outs = torch.cat(metapath_outs, dim=0)

        #h = torch.sum(metapath_outs, dim=0)#输出与注意力权重相乘 进行加权求和h = torch.sum(beta * metapath_outs, dim=0)
        h = torch.sum(beta * metapath_outs, dim=0)#
        return h, beta # #节点特征表示 h，以及用于注意力机制的权重 beta
#HNEMA_ctr_ntype_specific类是一个用于处理具有不同 metapath 的输入数据的模型，它包含了多个 HNEMA_metapath_specific 类的实例，并在这些实例的基础上实现了对不同 metapath 的加权汇总。


# class HNEMA_ctr_ntype_specific_transformer(nn.Module):
#     def __init__(self,
#                  num_metapaths,
#                  etypes_list,
#                  out_dim,
#                  num_heads,
#                  attn_vec_dim,
#                  rnn_type='transformer',
#                  attn_drop=0.5,
#                  use_minibatch=False,
#                  attn_switch=False,
#                  rnn_concat=False):
#         super(HNEMA_ctr_ntype_specific_transformer, self).__init__()
#         self.out_dim = out_dim
#         self.num_heads = num_heads
#         self.use_minibatch = use_minibatch
#         self.rnn_concat = rnn_concat
#
#         # metapath-specific layers
#         self.metapath_layers = nn.ModuleList()
#         for i in range(num_metapaths):
#             self.metapath_layers.append(HNEMA_metapath_specific(etypes_list[i],
#                                                                 out_dim,
#                                                                 num_heads,
#                                                                 rnn_type,
#                                                                 attn_drop=attn_drop,
#                                                                 use_minibatch=use_minibatch,
#                                                                 attn_switch=attn_switch,
#                                                                 rnn_concat=rnn_concat))
#
#         # metapath-level attention
#         # note that the acutal input dimension should consider the number of heads
#         # as multiple head outputs are concatenated together
#         if (self.rnn_concat == True):
#             self.fc1 = nn.Linear(out_dim * num_heads * 2, attn_vec_dim, bias=True)
#             self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
#             self.metapath_fuse = Attention_fuse(out_dim * num_heads * 2, attn_vec_dim)
#         else:
#             self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
#             self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)
#             self.metapath_fuse = Attention_fuse(out_dim * num_heads, attn_vec_dim)
#
#         # weight initialization
#         nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
#         nn.init.xavier_normal_(self.fc2.weight, gain=1.414)
#
#     def forward(self, inputs):
#         if self.use_minibatch:
#             g_list, features, type_mask, edge_metapath_indices_list, target_idx_list = inputs
#
#             # metapath-specific layers
#             if self.rnn_concat == True:
#                 metapath_outs = [
#                     F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,
#                                                                                                            self.num_heads * self.out_dim * 2))
#                     for g, edge_metapath_indices, target_idx, metapath_layer in
#                     zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
#             else:
#                 metapath_outs = [
#                     F.elu(metapath_layer((g, features, type_mask, edge_metapath_indices, target_idx)).view(-1,
#                                                                                                            self.num_heads * self.out_dim))
#                     for g, edge_metapath_indices, target_idx, metapath_layer in
#                     zip(g_list, edge_metapath_indices_list, target_idx_list, self.metapath_layers)]
#
#
#         else:
#             g_list, features, type_mask, edge_metapath_indices_list = inputs
#
#             # metapath-specific layers
#             metapath_outs = [F.elu(
#                 metapath_layer((g, features, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim))
#                 for g, edge_metapath_indices, metapath_layer in
#                 zip(g_list, edge_metapath_indices_list, self.metapath_layers)]
#
#         metapath_outs = torch.tensor([item.cpu().detach().numpy() for item in metapath_outs]).cuda()
#         # add non-linearity to fusing node features of different view
#         # Q = metapath_outs, K = metapath_outs, V = metapath_outs
#         h = self.metapath_fuse(metapath_outs, metapath_outs, metapath_outs)
#         return h


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Attention_fuse(nn.Module):
    def __init__(self, dim_model, attn_vec_dim):
        super(Attention_fuse, self).__init__()
        self.linears = clones(nn.Linear(dim_model, attn_vec_dim), 2)

        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight, gain=1.414)

    def forward(self, query, key, value):
        metapath_num = query.size(0)
        feature_dim = query.size(-1)
        # flatten the sequence dim of query and key
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        query = query.reshape(metapath_num, -1)
        key = key.reshape(metapath_num, -1)
        value = value.reshape(metapath_num, -1)
        attention = torch.mm(query, torch.t(key))
        d_k = torch.tensor(key.size(-1), dtype=torch.float32)
        attention = torch.div(attention, torch.sqrt(d_k))
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, value).reshape(metapath_num, -1, feature_dim)
        output = torch.mean(output, dim=0)
        # print(attention)
        return output

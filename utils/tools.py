import torch
import dgl
import numpy as np

def parse_adjlist(adjlist, edge_metapath_indices, samples=None, exclude=None, offset=None, mode=None):
    edges = []   #解析邻接列表数据，并返回边的列表、节点数量以及节点的映射关系  edge_metapath_indices 参数用于存储与每个节点相关的边的元路径索引
    nodes = set()
    result_indices = []#存结果的索引

    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' '))) #将当前行拆分为整数，并存储在 row_parsed 中。
        nodes.add(row_parsed[0])#将当前行的第一个节点添加到节点集合中
        # if meta-path based neighbors exist
        if len(row_parsed) > 1:#长度大于 1，则表示存在基于元路径的邻居节点
            # sampling neighbors
            if samples is None:#检查是否需要对邻居进行采样
                if exclude is not None:#检查是否有需要排除的邻居
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                u1, a1, u2, a2 in indices[:, [0, 1, -1, -2]]]
                        #mask= 根据给定的排除条件生成一个布尔蒙版，用于筛选邻居
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                a1, u1, a2, u2 in indices[:, [0, 1, -1, -2]]]
                    neighbors = np.array(row_parsed[1:])[mask]
                    result_indices.append(indices[mask])
                else:
                    neighbors = row_parsed[1:]
                    result_indices.append(indices)#将当前行的元路径索引添加到结果索引列表中。
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = [] #根据邻居节点出现的频率计算采样概率。
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()

                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))

                if exclude is not None:#需要对邻居节点筛选
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]#标记保留的邻居节点
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for
                                a1, u1, a2, u2 in indices[sampled_idx][:, [0, 1, -1, -2]]]

                    neighbors = np.array([row_parsed[i + 1] for i in sampled_idx])[mask]
                    result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = [row_parsed[i + 1] for i in sampled_idx]
                    result_indices.append(indices[sampled_idx])

        # for the case that a node does not have any neighbors
        else:#处理没有邻居节点的情况：如果当前行只包含一个节点，则将其作为自身的邻居节点处理。将该节点的索引添加到 result_indices 中。
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            if mode == 1:
                indices += offset
            result_indices.append(indices)

        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))

    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    # use mapping to transform edge ids
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)

    return edges, result_indices, len(nodes), mapping

#解析mini批次数据  Iteration 00000, Iteration 00100,Iteration 00200 ...：表示当前epoch中的迭代次数，每个iteration处理一个batch的数据。
def parse_minibatch(adjlists_ua, edge_metapath_indices_list_ua, drug_target_batch, device, samples=None,
                        use_masks=None, offset=None):
    # first parameter: meta-path based neighbors for each drug node
    # second parameter: nodes in every meta-path instance stored using relative ids
    # third parameter: node ids of drug-drug pairs in current batch
    g_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    # the loop for iterating the drug node and target node
    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_ua, edge_metapath_indices_list_ua)):
        # the loop for iterating every metapath of one type of node
        # the order of adjlist and indices are the same
         for adjlist, indices, use_mask in zip(adjlists, edge_metapath_indices_list, use_masks[mode]):
            if use_mask:#是否对特定节点类型的邻接列表进行筛选或采样。
                # samples=100
                #[adjlist[row[mode]] if mode < len(row) else None for row in drug_target_batch]
                edges, result_indices, num_nodes, mapping = parse_adjlist(
                    [adjlist[row[mode]] for row in drug_target_batch],
                    [indices[row[mode]] for row in drug_target_batch], samples, drug_target_batch, offset, mode)

            else:
                #[adjlist[row[mode]] if mode < len(row) else None for row in drug_target_batch]
                edges, result_indices, num_nodes, mapping = parse_adjlist(
                    [adjlist[row[mode]] for row in drug_target_batch],
                    [indices[row[mode]] for row in drug_target_batch], samples, offset=offset, mode=mode)


            # Multigraph means that there can be multiple edges between two nodes.
            # Multigraphs are graphs that can have multiple (directed) edges between the same pair of nodes, including self loops. For instance, two authors can coauthor a paper in different years, resulting in edges with different features.
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(num_nodes)

            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))

                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)

            else:
                result_indices = torch.LongTensor(result_indices).to(device)

            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in drug_target_batch]))
#遍历不同的节点类型，然后对每种节点类型的每个元路径进行处理。对于每个元路径，它根据参数进行筛选或采样，然后将结果转换为DGL图，并将处理后的结果存储在列表中。最后，它返回了处理后的图列表、结果索引列表和索引批次映射列表。
    # print(g_lists,len(g_lists))
    # print(result_indices_lists,len(result_indices_lists))
    # print(idx_batch_mapped_lists,len(idx_batch_mapped_lists))
    return g_lists, result_indices_lists, idx_batch_mapped_lists


class index_generator:#按批次
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0#初始化迭代计数器为0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):#返回总的迭代次数，即数据集中的样本数量除以批次大小
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):#返回剩余的迭代次数，即总迭代次数减去当前迭代次数
        return self.num_iterations() - self.iter_counter

    def reset(self):#重置迭代器状态，包括将迭代计数器设置为0和（如果指定了）重新打乱索引数组的顺序。
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
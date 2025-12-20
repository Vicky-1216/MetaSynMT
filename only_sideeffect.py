import time
import argparse
import torch
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
import numpy as np
from model.Auxiliary_networks import therapeutic_effect_DNN_predictor, side_effect_DNN_predictor2
from model.link_prediction import HNEMA_link_prediction
from utils.pytorchtools import EarlyStopping
from utils.data import load_HNEMA_DDI_data_te, load_HNEMA_DDI_data_se
from utils.tools import index_generator, parse_minibatch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score,recall_score
import random
import itertools
from model.Auxiliary_networks import AutomaticWeightedLoss
import pandas as pd
import scipy.stats
import logging
import copy
import csv
from sklearn import metrics


## fix random seed
random_seed = 1024
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
print('random_seed:', random_seed)
# import random
# random.seed()  # 使用系统时间作为种子
# random_seed = random.randint(0, 2**31 - 1)
# print('random_seed:', random_seed)
# some overall fixed parameters
# drug/target/cell line
num_ntype = 2#不用疾病
# for the main_net
dropout_rate = 0.3
lr = 0.0005 #原来是0.005
weight_decay = 0.001

# the aim of use_masks is to mask drug-drug pairs occurring in the batch, which contains these pairs as the known samples
use_masks = [[False, False, False, True],
             [False, False, False, True]]
# while in val/test set, such masks are not needed
no_masks = [[False] * 4, [False] * 4]

# total numbers of drug and target nodes
num_drug = 232
num_target = 3871

# involved_metapaths = [
#     [(0, 1, 0), (0, 1, 1, 0), (0, 1, 1, 1, 0), (0, 1, 1,1, 1, 0),(0, 'se', 0)]]#原来是te  (0, 1, 1, 0), (0, 1, 1, 1, 0),

# for the case that just load model for test
only_test = False

# the type of synergy score to be predicted
# S_mean, synergy_zip, synergy_loewe, synergy_hsa, synergy_bliss (corresponding to 0,1,2,3,4, respectively)
predicted_se_type = 0 #这里的标签直接用label

def run_model_HNEMA_DDI(root_prefix, hidden_dim_main, num_heads_main, attnvec_dim_main, rnn_type_main,
                        num_epochs, patience, batch_size, neighbor_samples, repeat, attn_switch_main, rnn_concat_main,#hidden_dim_aux,
                         layer_list, pred_in_dropout, pred_out_dropout, args):
    # root_prefix, hidden_dim_main, rnn_type_main,
    # num_epochs, patience, batch_size, neighbor_samples, repeat, rnn_concat_main,  # hidden_dim_aux,
    # layer_list, pred_in_dropout, pred_out_dropout, args
    print('output_concat, hidden_dim_aux, rnn_type_main:',rnn_type_main)##
##    adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, name2id_dict, train_val_test_drug_drug_samples, train_val_test_drug_drug_labels, all_drug_morgan, cellline_expression = load_HNEMA_DDI_data_te(root_prefix)
    adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, name2id_dict, train_val_test_drug_drug_samples, train_val_test_drug_drug_labels = load_HNEMA_DDI_data_se(root_prefix)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    features_list = []
    in_dims = []

    # based on type mask, to generate one-hot encoding for each type of nodes (drug/target/cell line) in the heterogeneous network
    # for i in range(num_ntype):
    #     dim = (type_mask == i).sum()
    #     in_dims.append(dim)
    #     indices = np.vstack((np.arange(dim), np.arange(dim)))
    #     indices = torch.LongTensor(indices)
    #     values = torch.FloatTensor(np.ones(dim))
    #     features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))

    for i in range(num_ntype):
        dim = (type_mask == i).sum()
        in_dims.append(dim)

        # 方法5：保持稀疏结构但值随机
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        # 只将对角线值改为随机值
        values = torch.rand(dim)  # 随机值代替全1

        features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))

    # ECFP6 of drugs
##    morgan_values = all_drug_morgan.data
##    morgan_indices = np.vstack((all_drug_morgan.row, all_drug_morgan.col))
##    i = torch.LongTensor(morgan_indices)
##    v = torch.FloatTensor(morgan_values)
##    shape = all_drug_morgan.shape
##    all_drug_morgan = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().to(device)

    train_drug_drug_samples = train_val_test_drug_drug_samples['train_drug_drug_samples']

    # scaler = MinMaxScaler()
    train_se_temp_labels = train_val_test_drug_drug_labels['train_se_labels'][:, predicted_se_type].reshape(-1,1)
    # scaler.fit(train_te_temp_labels)
    # train_te_temp_labels = scaler.transform(train_te_temp_labels)
    train_se_labels = torch.tensor(train_se_temp_labels, dtype=torch.float32).to(device)

    # an extra test about exchanging the val and test set
    val_drug_drug_samples = train_val_test_drug_drug_samples['val_drug_drug_samples']
    test_drug_drug_samples = train_val_test_drug_drug_samples['test_drug_drug_samples']

    val_se_temp_labels = train_val_test_drug_drug_labels['val_se_labels'][:, predicted_se_type].reshape(-1, 1)
    # test_te_temp_labels = scaler.transform(test_te_temp_labels)
    val_se_labels = torch.tensor(val_se_temp_labels,dtype=torch.float32).to(device)

    test_se_temp_labels = train_val_test_drug_drug_labels['test_se_labels'][:, predicted_se_type].reshape(-1, 1)
    # val_te_temp_labels = scaler.transform(val_te_temp_labels)
    test_se_labels = torch.tensor(test_se_temp_labels,dtype=torch.float32).to(device)

##    se_symbol2id_dict = name2id_dict[-2]
##    disease2id_dict = name2id_dict[-3]
    ##disease2id_dict = name2id_dict[-1]
    mse_list = []
    rmse_list = []
    mae_list = []
    pearson_list = []

    VAL_L0SS=[]
    for _ in range(repeat):
        main_net = HNEMA_link_prediction(
            [6], in_dims[:-1], hidden_dim_main, hidden_dim_main, num_heads_main, attnvec_dim_main, rnn_type_main,
            dropout_rate, attn_switch_main, rnn_concat_main, args)
        # [5], in_dims[:-1], hidden_dim_main, hidden_dim_main, rnn_type_main,
        # dropout_rate, rnn_concat_main, args
        main_net.to(device)
        print(main_net)##

        se_layer_list = copy.deepcopy(layer_list)#[2048 1024 512]
        se_layer_list.append(1)
        print('The hidden unit number for each layer in SE prediction:', se_layer_list)

        se_net = side_effect_DNN_predictor2(hidden_dim_main, se_layer_list,##hidden_dim_aux, output_concat,
                                            pred_out_dropout, pred_in_dropout)
                                       ##         hidden_dim_main + all_drug_morgan.shape[1], hidden_dim_aux, te_layer_list,
                                       ##         output_concat, len(se_symbol2id_dict), pred_out_dropout, pred_in_dropout)
        se_net.to(device)
        sigmoid = torch.nn.Sigmoid()##
        print('se_net structure:', se_net)
        # optimizer = torch.optim.SGD(
        optimizer = torch.optim.Adam(
            itertools.chain(main_net.parameters(), se_net.parameters()),
            lr=lr, weight_decay=weight_decay)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        main_net.train()
        se_net.train()

        tot_params1 = sum([np.prod(p.size()) for p in main_net.parameters()])
        print(f"Total number of parameters in main_net: {tot_params1}")
        tot_params2 = sum([np.prod(p.size()) for p in se_net.parameters()])
        print(f"Total number of parameters in te_net: {tot_params2}")
        print(f"Total number of parameters in Muthene: {tot_params1 + tot_params2}")

        if only_test == True:
            temp_prefix = './data/data4training_model/checkpoint/'
            # change it to your trained model
            model_save_path = temp_prefix + 'checkpoint.pt'
        else:
            model_save_path = root_prefix + 'checkpoint/checkpoint_{}.pt'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=model_save_path)
        # three lists keeping the time of different training phases
        dur1 = []  # data processing before feeding data in an iteration
        dur2 = []  # the training time for an iteration
        dur3 = []  # the time to use grad to update parameters of the model

        train_sample_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_drug_drug_samples))
        # reason for batch_size=batch_size//2: to generate the drug-drug pairs with the opposite drug order in val/test phases
        val_sample_idx_generator = index_generator(batch_size=batch_size//2, num_data=len(val_drug_drug_samples), shuffle=False)
        test_sample_idx_generator = index_generator(batch_size=batch_size//2, num_data=len(test_drug_drug_samples), shuffle=False)

        ##te_criterion = torch.nn.MSELoss(reduction='mean')回归问题
        se_criterion = torch.nn.BCELoss(reduction='mean')
        ##se_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')#输出可以不是0-1

        print('total epoch number is:',num_epochs)
        if only_test == False:
            for epoch in range(num_epochs):
                t_start = time.time()
                main_net.train()
                se_net.train()
                print('train_sample_idx_generator.num: ', train_sample_idx_generator.num_iterations())##
                for iteration in range(train_sample_idx_generator.num_iterations()):
                    t0 = time.time()
                    train_sample_idx_batch = train_sample_idx_generator.next()
                    train_sample_idx_batch.sort()

                    train_drug_drug_batch = train_drug_drug_samples[train_sample_idx_batch].tolist()
                    train_se_labels_batch = train_se_labels[train_sample_idx_batch]

                    train_drug_drug_idx = (np.array(train_drug_drug_batch).astype(int)).tolist()
                    ## train_drug_drug_idx = (np.array(train_drug_drug_batch)[:, :-1].astype(int)).tolist()##

                    train_g_lists, train_indices_lists, train_idx_batch_mapped_lists = parse_minibatch(adjlists_ua, edge_metapath_indices_list_ua, train_drug_drug_idx, device, neighbor_samples, use_masks, num_drug)

                    t1 = time.time()
                    dur1.append(t1 - t0)

                    [row_drug_embedding, col_drug_embedding], _,[row_drug_atten, col_drug_atten]  = main_net((train_g_lists, features_list, type_mask[:num_drug + num_target], train_indices_lists, train_idx_batch_mapped_lists))

                    train_drug_drug_idx = torch.tensor(train_drug_drug_idx, dtype=torch.int64).to(device)
                    ##train_cellline_idx = torch.tensor(train_cellline_idx, dtype=torch.int64).to(device)
                    row_drug_batch, col_drug_batch = train_drug_drug_idx[:, 0], train_drug_drug_idx[:, 1]
      ##              row_drug_struc_embedding, col_drug_struc_embedding = all_drug_morgan[row_drug_batch], all_drug_morgan[col_drug_batch]

      ##              row_drug_composite_embedding = torch.cat((row_drug_embedding, row_drug_struc_embedding), axis=1)
      ##              col_drug_composite_embedding = torch.cat((col_drug_embedding, col_drug_struc_embedding), axis=1)
                    row_drug_composite_embedding = row_drug_embedding
                    col_drug_composite_embedding = col_drug_embedding

                    ##se_output = se_net(row_drug_composite_embedding, col_drug_composite_embedding)
                    se_output = sigmoid(se_net(row_drug_composite_embedding, col_drug_composite_embedding)) #, train_cellline_idx))##用了sigmoid

                    se_loss = se_criterion(se_output, train_se_labels_batch)
                    train_total_loss_batch = se_loss#没有AE的总共损失

                    t2 = time.time()
                    dur2.append(t2 - t1)
                    # autograd
                    optimizer.zero_grad()
                    train_total_loss_batch.backward()
                    # clip_grad_norm_(itertools.chain(main_net.parameters(), drug_net.parameters(), te_net.parameters(), se_net.parameters()), max_norm=10, norm_type=2)
                    optimizer.step()
                    t3 = time.time()
                    dur3.append(t3 - t2)
                    if iteration % 10 == 0:#
                        print(
                            'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                                epoch, iteration, train_total_loss_batch.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))

                # model evaluation
                main_net.eval()
                se_net.eval()
                val_te_loss, val_total_loss=[],[]
                with torch.no_grad():
                    for iteration in range(val_sample_idx_generator.num_iterations()):
                        val_sample_idx_batch = val_sample_idx_generator.next()
                        val_drug_drug_batch = val_drug_drug_samples[val_sample_idx_batch]
                        # print("Number of columns:", val_drug_drug_batch.shape[1])
                        # for generating drug-drug pairs with the opposite drug order
                        # val_drug_drug_batch_ = val_drug_drug_batch[:, [1, 0]]
                        val_drug_drug_batch_ = val_drug_drug_batch[:, [1, 0, 2]]
                        # print("Number of columns:", val_drug_drug_batch_.shape[1])
                        # print("Shape of val_drug_drug_batch:", val_drug_drug_batch.shape)
                        # print("Shape of val_drug_drug_batch_:", val_drug_drug_batch_.shape)


                        val_drug_drug_batch_combined = np.concatenate([val_drug_drug_batch,val_drug_drug_batch_],axis=0).tolist()

                        val_se_labels_batch = val_se_labels[val_sample_idx_batch]

                        val_drug_drug_idx = (np.array(val_drug_drug_batch_combined).astype(int)).tolist()
                        ##val_drug_drug_idx = (np.array(val_drug_drug_batch_combined)[:, :-1].astype(int)).tolist()
                        ##val_cellline_symbol = (np.array(val_drug_drug_batch_combined)[:, -1]).tolist()
                        ##val_cellline_idx = [disease2id_dict[i] for i in val_cellline_symbol]

                        val_g_lists, val_indices_lists, val_idx_batch_mapped_lists = parse_minibatch(adjlists_ua, edge_metapath_indices_list_ua, val_drug_drug_idx, device, neighbor_samples, no_masks, num_drug)

                        [row_drug_embedding, col_drug_embedding], _,[row_drug_atten, col_drug_atten]  = main_net((val_g_lists, features_list, type_mask[:num_drug + num_target], val_indices_lists, val_idx_batch_mapped_lists))

                        val_drug_drug_idx = torch.tensor(val_drug_drug_idx, dtype=torch.int64).to(device)
                        ##val_cellline_idx = torch.tensor(val_cellline_idx, dtype=torch.int64).to(device)
                        row_drug_batch, col_drug_batch = val_drug_drug_idx[:, 0], val_drug_drug_idx[:, 1]
          ##            row_drug_struc_embedding, col_drug_struc_embedding = all_drug_morgan[row_drug_batch], all_drug_morgan[col_drug_batch]

          ##              row_drug_composite_embedding = torch.cat((row_drug_embedding, row_drug_struc_embedding), axis=1)
          ##              col_drug_composite_embedding = torch.cat((col_drug_embedding, col_drug_struc_embedding), axis=1)
                        row_drug_composite_embedding = row_drug_embedding
                        col_drug_composite_embedding = col_drug_embedding

                        ##se_output = se_net(row_drug_composite_embedding, col_drug_composite_embedding)
                        se_output = sigmoid(se_net(row_drug_composite_embedding, col_drug_composite_embedding))
                        # calculate the averaging results of the drug pairs with the opposite drug order
                        se_output = (se_output[:se_output.shape[0]//2,:] + se_output[se_output.shape[0]//2:,:])/2
                        se_loss = se_criterion(se_output, val_se_labels_batch)
                        val_total_loss.append(se_loss)

                    val_total_loss=torch.mean(torch.tensor(val_total_loss))
                    VAL_L0SS.append(val_total_loss.item())
                t_end = time.time()
                print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                    epoch, val_total_loss.item(), t_end - t_start))

                scheduler.step()
                early_stopping(val_total_loss,
                               {
                                   'main_net': main_net.state_dict(),
                                   'se_net': se_net.state_dict()
                               })
                if early_stopping.early_stop:
                    print('Early stopping based on the validation loss!')
                    break

        # model test
        print('The name of loaded model is:', model_save_path)
        checkpoint = torch.load(model_save_path)
        main_net.load_state_dict(checkpoint['main_net'])
        se_net.load_state_dict(checkpoint['se_net'])

        main_net.eval()
        se_net.eval()
        test_se_results = []
        test_se_label_list = []
        with torch.no_grad():
            for iteration in range(test_sample_idx_generator.num_iterations()):
                test_sample_idx_batch = test_sample_idx_generator.next()
                test_drug_drug_batch = test_drug_drug_samples[test_sample_idx_batch]
                #test_drug_drug_batch_ = test_drug_drug_batch[:, [1, 0]]
                test_drug_drug_batch_ = test_drug_drug_batch[:,[1,0,2]]

                test_drug_drug_idx_spec = (np.array(test_drug_drug_batch).astype(int)).tolist()
                ##test_drug_drug_idx_spec = (np.array(test_drug_drug_batch)[:, :-1].astype(int)).tolist()
                test_drug_drug_batch_combined = np.concatenate([test_drug_drug_batch,test_drug_drug_batch_],axis=0).tolist()

                test_se_labels_batch = test_se_labels[test_sample_idx_batch]
                test_drug_drug_idx = (np.array(test_drug_drug_batch_combined).astype(int)).tolist()

                test_g_lists, test_indices_lists, test_idx_batch_mapped_lists = parse_minibatch(
                    adjlists_ua, edge_metapath_indices_list_ua, test_drug_drug_idx, device, neighbor_samples,
                    no_masks, num_drug)

                # [row_drug_embedding, col_drug_embedding], _ = main_net((test_g_lists, features_list, type_mask[:num_drug + num_target], test_indices_lists, test_idx_batch_mapped_lists))
                [row_drug_embedding, col_drug_embedding], _, [row_drug_atten, col_drug_atten] = main_net((test_g_lists,
                                                                                                          features_list,
                                                                                                          type_mask[
                                                                                                          :num_drug + num_target],
                                                                                                          test_indices_lists,
                                                                                                          test_idx_batch_mapped_lists))
                test_drug_drug_idx = torch.tensor(test_drug_drug_idx, dtype=torch.int64).to(device)
                test_drug_drug_idx_spec = torch.tensor(test_drug_drug_idx_spec, dtype=torch.int64).to(device)
                row_drug_batch, col_drug_batch = test_drug_drug_idx[:, 0], test_drug_drug_idx[:, 1]
  ##              row_drug_struc_embedding, col_drug_struc_embedding = all_drug_morgan[row_drug_batch], all_drug_morgan[col_drug_batch]

  ##              row_drug_composite_embedding = torch.cat((row_drug_embedding, row_drug_struc_embedding), axis=1)
  ##              col_drug_composite_embedding = torch.cat((col_drug_embedding, col_drug_struc_embedding), axis=1)
                row_drug_composite_embedding = row_drug_embedding
                col_drug_composite_embedding = col_drug_embedding

                ##se_output = se_net(row_drug_composite_embedding, col_drug_composite_embedding)
                se_output = sigmoid(se_net(row_drug_composite_embedding, col_drug_composite_embedding)) #, test_cellline_idx))#用了sigmoid
                se_output = (se_output[:se_output.shape[0]//2,:] + se_output[se_output.shape[0]//2:,:])/2
                # print(test_drug_drug_idx_spec.shape)
                test_se_results.append(se_output)
                test_se_label_list.append(test_se_labels_batch)

                # 将药物对的ID信息添加到te_output和test_te_labels_batch中
                ## se_output = torch.cat((se_output, test_drug_drug_idx_spec), dim=1)
                ## test_se_labels_batch = torch.cat((test_se_labels_batch, test_drug_drug_idx_spec), dim=1)

                test_se_results.append(se_output)
                test_se_label_list.append(test_se_labels_batch)

            test_se_results = torch.cat(test_se_results)
            test_se_results = test_se_results.cpu().numpy()
            # test_te_results = scaler.inverse_transform(test_te_results)

            test_se_label_list = torch.cat(test_se_label_list)
            test_se_label_list = test_se_label_list.cpu().numpy()
            # test_te_label_list = scaler.inverse_transform(test_te_label_list)

       ## print('test_se_results:', test_se_results)
       ## print('test_se_label_list:', test_se_label_list)
       ## with open('D:/daima/Muthene-main/Muthene_dataset/fold1/test_se_results.csv', 'w', newline='') as csv_file:
       ##     writer = csv.writer(csv_file)
       ##     writer.writerows(test_se_results)
       ## with open('D:/daima/Muthene-main/Muthene_dataset/fold1/test_se_label_list.csv', 'w', newline='') as csv_file:
       ##     writer = csv.writer(csv_file)
       ##     writer.writerows(test_se_label_list)
        # print('test_se_results:', test_se_results)
        # print('test_se_label_list:', test_se_label_list)
        with open('E:/Muthene-main/echino_dataset/side/fold5/test_se_safe_results.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(test_se_results)
        # with open('C:/Users/Administrator/Desktop/Muthene-main/echino_dataset/side/fold2/test_se_label_list.csv', 'w', newline='') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerows(test_se_label_list)
        # print('the size of test_se_results:', test_se_results.shape)
        # print('the size of test_se_label_list:', test_se_label_list.shape)

        # 计算 ROC AUC 和 PR AUC
        roc_auc = roc_auc_score(test_se_label_list, test_se_results)
        pr_auc = average_precision_score(test_se_label_list, test_se_results)
        precision = precision_score(test_se_label_list, test_se_results.round())
        accuracy = accuracy_score(test_se_label_list, test_se_results.round())
        f1 = metrics.f1_score(test_se_label_list, test_se_results.round())
        recall = recall_score(test_se_label_list, test_se_results.round())
        # 打印结果
        print('ROC AUC =', roc_auc)
        print('PR AUC =', pr_auc)
        print('ACC =', accuracy)
        print('PREC =', precision)
        print('RECALL =', recall)
        print('F1 =', f1)


    pd.DataFrame(VAL_L0SS, columns=['VAL_LOSS']).to_csv(
        root_prefix+'checkpoint/VAL_LOSS.csv')


if __name__ == '__main__':
    ##seed=1024
    # part1 (for meta-path embedding generation)
    ap = argparse.ArgumentParser(description='Muthene SE module variant testing for drug-drug link prediction')
    ap.add_argument('--root-prefix', type=str,
                    default='E:/Muthene-main/echino_dataset/side/fold3/', # the folder to store the model input for current independent repeat
                    help='root from which to read the original input files')
    ap.add_argument('--hidden-dim-main', type=int, default=64,
                    help='Dimension of the node hidden state in the main model. Default is 64.')
    ap.add_argument('--num-heads-main', type=int, default=8,
                    help='Number of the attention heads in the main model. Default is 8.')
    ap.add_argument('--attnvec-dim-main', type=int, default=128,
                    help='Dimension of the attention vector in the main model. Default is 128.')
    ap.add_argument('--rnn-type-main', default='rnn',
                    help='Type of the aggregator in the main model. Default is rnn.')
    ap.add_argument('--epoch', type=int, default=30, help='Number of epochs. Default is 50.')
    ap.add_argument('--patience', type=int, default=8, help='Patience. Default is 10.')##原来是8
    ap.add_argument('--batch-size', type=int, default=16,##原来是32
                    help='Batch size. Please choose an odd value, because of the way of calculating val/test labels of our model. Default is 32.')
    ap.add_argument('--samples', type=int, default=100, #采样的邻居节点数 原来是100
                    help='Number of neighbors sampled in the parse function of main model. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    # if it is set to False, the GAT layer will ignore the feature of the central node itself
    ap.add_argument('--attn-switch-main', default=True,
                    help='whether need to consider the feature of the central node when using GAT layer in the main model')
    ap.add_argument('--rnn-concat-main', default=False,##原来是false
                    help='whether need to concat the feature extracted from rnn with the embedding from GAT layer in the main model')
    # part2
    # ap.add_argument('--hidden-dim-aux', type=int, default=64,
    #                 help='Dimension of generated cell line embeddings. Default is 64.')
    ap.add_argument('--layer-list', default=[2048, 1024, 512], ##default = [2048, 1024, 512]
                    help='layer neuron units list for the DNN TE predictor.')
    ap.add_argument('--pred_in_dropout', type=float, default=0.2,
                    help='The input dropout rate of the DNN TE predictor')
    ap.add_argument('--pred_out_dropout', type=float, default=0.5,
                    help='The output dropout rate of the DNN TE predictor')
    # ap.add_argument('--output_concat', default=False,
    #                 help='Whether put the adverse effect output into therapeutiec effect prediction')

    args = ap.parse_args()
    run_model_HNEMA_DDI(args.root_prefix, args.hidden_dim_main, args.num_heads_main, args.attnvec_dim_main, args.rnn_type_main, args.epoch,
                        args.patience, args.batch_size, args.samples, args.repeat, args.attn_switch_main, args.rnn_concat_main, #args.hidden_dim_aux,
                        args.layer_list, args.pred_in_dropout, args.pred_out_dropout,  args)

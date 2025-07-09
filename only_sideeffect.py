import time
import argparse
import torch
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
import numpy as np
from model.Predictor import therapeutic_effect_predictor, side_effect_predictor2
from model.link_prediction import link_prediction
from utils.pytorchtools import EarlyStopping
from utils.data import load_data_te, load_data_se
from utils.tools import index_generator, parse_minibatch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
import random
import itertools
from model.Predictor import AutomaticWeightedLoss
import pandas as pd
import scipy.stats
import logging
import copy
import csv
from sklearn import metrics
import os
import pickle


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
#print('random_seed:', random_seed)
# np.random.seed(42)
# random.seed(42) #种子是42

# import random
# random.seed()  # 使用系统时间作为种子
# random_seed = random.randint(0, 2**31 - 1)
# print('random_seed:', random_seed)
# some overall fixed parameters
# drug/target/cell line
num_ntype = 2#不用疾病
# for the main_net
dropout_rate = 0.5
lr = 0.0005 #原来是0.005
weight_decay = 0.001#原来是0.001

# the aim of use_masks is to mask drug-drug pairs occurring in the batch, which contains these pairs as the known samples
use_masks = [[False, False, False, True],
             [False, False, False, True]]
# while in val/test set, such masks are not needed
no_masks = [[False] * 4, [False] * 4]

# total numbers of drug and target nodes
num_drug = 232
num_target = 3871

involved_metapaths = [
    [(0, 1, 0), (0, 1, 1, 0), (0, 1, 1, 1, 0), (0, 'se', 0)]]

# for the case that just load model for test
only_test = True

# the type of synergy score to be predicted
predicted_se_type = 0 #这里的标签直接用label

def run_model(root_prefix, hidden_dim_main, num_heads_main, attnvec_dim_main, rnn_type_main,
                        num_epochs, patience, batch_size, neighbor_samples, repeat, attn_switch_main, rnn_concat_main,#hidden_dim_aux,
                         layer_list, pred_in_dropout, pred_out_dropout, args):

    #print('output_concat, hidden_dim_aux, rnn_type_main:',rnn_type_main)##
    adjlists_ua, edge_metapath_indices_list_ua, adjM, type_mask, name2id_dict, train_val_test_drug_drug_samples, train_val_test_drug_drug_labels = load_data_se(root_prefix)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    features_list = []
    in_dims = []

    # based on type mask, to generate one-hot encoding for each type of nodes (drug/target/disese) in the heterogeneous network
    for i in range(num_ntype):
        dim = (type_mask == i).sum()
        in_dims.append(dim)
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))

    train_drug_drug_samples = train_val_test_drug_drug_samples['train_drug_drug_samples']

    # scaler = MinMaxScaler()
    train_se_temp_labels = train_val_test_drug_drug_labels['train_se_labels'][:, predicted_se_type].reshape(-1,1)
    # scaler.fit(train_te_temp_labels)
    # train_te_temp_labels = scaler.transform(train_te_temp_labels)
    train_se_labels = torch.tensor(train_se_temp_labels, dtype=torch.float32).to(device)

    # an extra test about exchanging the val and test set
    val_drug_drug_samples = train_val_test_drug_drug_samples['val_drug_drug_samples']
    #test_drug_drug_samples = train_val_test_drug_drug_samples['test_drug_drug_samples']

    val_se_temp_labels = train_val_test_drug_drug_labels['val_se_labels'][:, predicted_se_type].reshape(-1, 1)
    # test_te_temp_labels = scaler.transform(test_te_temp_labels)
    val_se_labels = torch.tensor(val_se_temp_labels,dtype=torch.float32).to(device)

    file_path = './static/MPSyn_Schistosomiasis.csv'
    # # file.save(file_path)
    df = pd.read_csv(os.path.join(file_path))
    with open('./static/syn/drug2absid_dict.pickle', 'rb') as f:
        drug_dict = pickle.load(f)
    # 定义一个函数，用于查找药物名称对应的序号
    def find_drug_code(name, drug_dict):
        return drug_dict.get(name, None)  # 如果药物名不在字典中，返回None

    # 应用这个函数到CSV文件的第一列和第二列
    df['Drug1'] = df['drug1'].apply(lambda x: find_drug_code(x, drug_dict))
    df['Drug2'] = df['drug2'].apply(lambda x: find_drug_code(x, drug_dict))

    # 将转换后的数据转换为列表
    column1_codes = df['Drug1'].tolist()
    column2_codes = df['Drug2'].tolist()

    # 将两个列表组合为一个列表的列表（二维列表）
    test_drug_drug_samples = list(zip(column1_codes, column2_codes))

    # 如果你需要将DataFrame转换为列表
    # test_drug_drug_samples = df.values.tolist()
    test_drug_drug_array = np.array(test_drug_drug_samples)
    test_drug_drug_samples_dict = {'test_drug_drug_samples': test_drug_drug_array}
    test_drug_drug_samples = test_drug_drug_samples_dict['test_drug_drug_samples']
    # test_se_temp_labels = train_val_test_drug_drug_labels['test_se_labels'][:, predicted_se_type].reshape(-1, 1)
    # # val_te_temp_labels = scaler.transform(val_te_temp_labels)
    # test_se_labels = torch.tensor(test_se_temp_labels,dtype=torch.float32).to(device)

##    se_symbol2id_dict = name2id_dict[-2]
##    cellline2id_dict = name2id_dict[-3]
    ##disease2id_dict = name2id_dict[-1]
    # mse_list = []
    # rmse_list = []
    # mae_list = []
    # pearson_list = []

    VAL_L0SS=[]
    for _ in range(repeat):
        main_net = link_prediction(
            [4], in_dims[:-1], hidden_dim_main, hidden_dim_main, num_heads_main, attnvec_dim_main, rnn_type_main,
            dropout_rate, attn_switch_main, rnn_concat_main, args)
        main_net.to(device)
        #print(main_net)##

        se_layer_list = copy.deepcopy(layer_list)#[2048 1024 512]
        se_layer_list.append(1)
        #print('The hidden unit number for each layer in SE prediction:', se_layer_list)
        se_net = side_effect_predictor2(hidden_dim_main, se_layer_list,##hidden_dim_aux, output_concat,
                                            pred_out_dropout, pred_in_dropout)

        se_net.to(device)
        sigmoid = torch.nn.Sigmoid()##
        #print('se_net structure:', se_net)
        # optimizer = torch.optim.SGD(
        optimizer = torch.optim.Adam(
            itertools.chain(main_net.parameters(), se_net.parameters()),
            lr=lr, weight_decay=weight_decay)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        main_net.train()
        se_net.train()

        tot_params1 = sum([np.prod(p.size()) for p in main_net.parameters()])
        tot_params2 = sum([np.prod(p.size()) for p in se_net.parameters()])
        #print(f"Total number of parameters in model: {tot_params1 + tot_params2}")#print(f"Total number of parameters in te_net: {tot_params2}") print(f"Total number of parameters in main_net: {tot_params1}")

        if only_test == True:
            temp_prefix = './best_model/checkpoint_side/'
            # change it to your trained model
            model_save_path = temp_prefix + 'checkpoint_side.pt'
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

        se_criterion = torch.nn.BCELoss(reduction='mean')
        ##se_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')#输出可以不是0-1

        #print('total epoch number is:',num_epochs)
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

                    train_g_lists, train_indices_lists, train_idx_batch_mapped_lists = parse_minibatch(adjlists_ua, edge_metapath_indices_list_ua, train_drug_drug_idx, device, neighbor_samples, use_masks, num_drug)

                    t1 = time.time()
                    dur1.append(t1 - t0)

                    #[row_drug_embedding, col_drug_embedding], _, [row_drug_atten, col_drug_atten] = main_net((train_g_lists, features_list, type_mask[:num_drug + num_target], train_indices_lists, train_idx_batch_mapped_lists))
                    [row_drug_embedding, col_drug_embedding], _, [row_drug_atten, col_drug_atten] = main_net((train_g_lists, features_list, type_mask[:num_drug + num_target], train_indices_lists, train_idx_batch_mapped_lists))

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
                        # for generating drug-drug pairs with the opposite drug order
                        val_drug_drug_batch_ = val_drug_drug_batch[:, [1, 0]]
                        ##val_drug_drug_batch_ = val_drug_drug_batch[:,[1,0,2]]
                        val_drug_drug_batch_combined = np.concatenate([val_drug_drug_batch,val_drug_drug_batch_],axis=0).tolist()

                        val_se_labels_batch = val_se_labels[val_sample_idx_batch]

                        val_drug_drug_idx = (np.array(val_drug_drug_batch_combined).astype(int)).tolist()


                        val_g_lists, val_indices_lists, val_idx_batch_mapped_lists = parse_minibatch(adjlists_ua, edge_metapath_indices_list_ua, val_drug_drug_idx, device, neighbor_samples, no_masks, num_drug)

                        [row_drug_embedding, col_drug_embedding], _, [row_drug_atten, col_drug_atten] = main_net((val_g_lists, features_list, type_mask[:num_drug + num_target], val_indices_lists, val_idx_batch_mapped_lists))

                        val_drug_drug_idx = torch.tensor(val_drug_drug_idx, dtype=torch.int64).to(device)

                        row_drug_batch, col_drug_batch = val_drug_drug_idx[:, 0], val_drug_drug_idx[:, 1]
          ##            row_drug_struc_embedding, col_drug_struc_embedding = all_drug_morgan[row_drug_batch], all_drug_morgan[col_drug_batch]

          ##              row_drug_composite_embedding = torch.cat((row_drug_embedding, row_drug_struc_embedding), axis=1)
          ##              col_drug_composite_embedding = torch.cat((col_drug_embedding, col_drug_struc_embedding), axis=1)
                        row_drug_composite_embedding = row_drug_embedding
                        col_drug_composite_embedding = col_drug_embedding

                        ##se_output = se_net(row_drug_composite_embedding, col_drug_composite_embedding)
                        se_output = sigmoid(se_net(row_drug_composite_embedding, col_drug_composite_embedding))#, val_cellline_idx))##用了sigmoid
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
        #print('The name of loaded model is:', model_save_path)
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
                test_drug_drug_batch_ = test_drug_drug_batch[:, [1, 0]]
                ##test_drug_drug_batch_ = test_drug_drug_batch[:,[1,0,2]]

                test_drug_drug_idx_spec = (np.array(test_drug_drug_batch).astype(int)).tolist()
                ##test_drug_drug_idx_spec = (np.array(test_drug_drug_batch)[:, :-1].astype(int)).tolist()
                test_drug_drug_batch_combined = np.concatenate([test_drug_drug_batch,test_drug_drug_batch_],axis=0).tolist()

                #test_se_labels_batch = test_se_labels[test_sample_idx_batch]
                test_drug_drug_idx = (np.array(test_drug_drug_batch_combined).astype(int)).tolist()

                test_g_lists, test_indices_lists, test_idx_batch_mapped_lists = parse_minibatch(
                    adjlists_ua, edge_metapath_indices_list_ua, test_drug_drug_idx, device, neighbor_samples,
                    no_masks, num_drug)

               # [row_drug_embedding, col_drug_embedding], _, [row_drug_atten, col_drug_atten] = main_net((test_g_lists, features_list, type_mask[:num_drug + num_target], test_indices_lists, test_idx_batch_mapped_lists))
                [row_drug_embedding, col_drug_embedding], _, [row_drug_atten, col_drug_atten] = main_net((test_g_lists, features_list, type_mask[:num_drug + num_target], test_indices_lists, test_idx_batch_mapped_lists))

                test_drug_drug_idx = torch.tensor(test_drug_drug_idx, dtype=torch.int64).to(device)
                test_drug_drug_idx_spec = torch.tensor(test_drug_drug_idx_spec, dtype=torch.int64).to(device)
                ##test_cellline_idx = torch.tensor(test_cellline_idx, dtype=torch.int64).to(device)
                row_drug_batch, col_drug_batch = test_drug_drug_idx[:, 0], test_drug_drug_idx[:, 1]
  ##              row_drug_struc_embedding, col_drug_struc_embedding = all_drug_morgan[row_drug_batch], all_drug_morgan[col_drug_batch]

  ##              row_drug_composite_embedding = torch.cat((row_drug_embedding, row_drug_struc_embedding), axis=1)
  ##              col_drug_composite_embedding = torch.cat((col_drug_embedding, col_drug_struc_embedding), axis=1)
                row_drug_composite_embedding = row_drug_embedding
                col_drug_composite_embedding = col_drug_embedding

                ##se_output = se_net(row_drug_composite_embedding, col_drug_composite_embedding)
                se_output = sigmoid(se_net(row_drug_composite_embedding, col_drug_composite_embedding)) #, test_cellline_idx))#用了sigmoid
                se_output = (se_output[:se_output.shape[0]//2,:] + se_output[se_output.shape[0]//2:,:])/2
                #print(test_drug_drug_idx_spec.shape)
        ##        print(test_drug_drug_idx_spec) #32乘2
        ##        print(te_output.shape) #16乘1
                test_se_results.append(se_output)
                #test_se_label_list.append(test_se_labels_batch)

                # 将药物对的ID信息添加到te_output和test_te_labels_batch中
                ## se_output = torch.cat((se_output, test_drug_drug_idx_spec), dim=1)
                ## test_se_labels_batch = torch.cat((test_se_labels_batch, test_drug_drug_idx_spec), dim=1)

            test_se_results = torch.cat(test_se_results)
            test_se_results = test_se_results.cpu().numpy()
            print(test_se_results)
            # test_te_results = scaler.inverse_transform(test_te_results)

            # test_se_label_list = torch.cat(test_se_label_list)
            # test_se_label_list = test_se_label_list.cpu().numpy()
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
        # with open('C:/Users/Administrator/Desktop/Muthene-main/echino_dataset/side/fold5/pingtai.csv', 'w', newline='') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerows(test_se_results)
        # with open('C:/Users/Administrator/Desktop/Muthene-main/echino_dataset/side/fold2/test_se_label_2repeatlist.csv', 'w', newline='') as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerows(test_se_label_list)
        # print('the size of test_se_results:', test_se_results.shape)
        # print('the size of test_se_label_list:', test_se_label_list.shape)


    # pd.DataFrame(VAL_L0SS, columns=['VAL_LOSS']).to_csv(
    #     root_prefix+'checkpoint/VAL_LOSS.csv')


if __name__ == '__main__':
    ##lr = 0.0001  # 原来是0.005 weight_decay = 0.001 原来0.001 dropout=0.5原0.5
    ##seed=1024
    # part1 (for meta-path embedding generation)
    ap = argparse.ArgumentParser(description='SE module variant testing for drug-drug link prediction')
    ap.add_argument('--root-prefix', type=str,
                    default='./static/side/', # the folder to store the model input for current independent repeat
                    help='root from which to read the original input files')
    ap.add_argument('--hidden-dim-main', type=int, default=64,
                    help='Dimension of the node hidden state in the main model. Default is 64')
    ap.add_argument('--num-heads-main', type=int, default=8,
                    help='Number of the attention heads in the main model. Default is 8.')
    ap.add_argument('--attnvec-dim-main', type=int, default=128,
                    help='Dimension of the attention vector in the main model. Default is 128.')
    ap.add_argument('--rnn-type-main', default='rnn',
                    help='Type of the aggregator in the main model. Default is rnn.')
    ap.add_argument('--epoch', type=int, default=30, help='Number of epochs. Default is 50.')
    ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 10.')##原来是8
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

    ap.add_argument('--layer-list', default=[2048, 1024, 512], ##default = [2048, 1024, 512]
                    help='layer neuron units list for the DNN TE predictor.')
    ap.add_argument('--pred_in_dropout', type=float, default=0.2,
                    help='The input dropout rate of the DNN TE predictor')
    ap.add_argument('--pred_out_dropout', type=float, default=0.5,
                    help='The output dropout rate of the DNN TE predictor')
    # ap.add_argument('--output_concat', default=False,
    #                 help='Whether put the adverse effect output into therapeutiec effect prediction')

    args = ap.parse_args()
    run_model(args.root_prefix, args.hidden_dim_main, args.num_heads_main, args.attnvec_dim_main, args.rnn_type_main, args.epoch,
                        args.patience, args.batch_size, args.samples, args.repeat, args.attn_switch_main, args.rnn_concat_main, #args.hidden_dim_aux,
                        args.layer_list, args.pred_in_dropout, args.pred_out_dropout,  args)

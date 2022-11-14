import data_pp_tools
import argparse
import pprint as pp
import cnn_lstm_model
import logging
import os
import time
import timeit
import torch
import tools


def main(args):
    cuda_gpu = int(args['cuda_gpu'])
    log_en = int(args['log_en'])
    epochs = int(args['epochs'])
    adam_lr = float(args['adam_lr'])
    batch_size = int(args['batch_size'])
    shuffle_data = int(args['shuffle_data'])

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # device = torch.device('cuda:'+str(cuda_gpu))
        # device = torch.device('cuda')
        device = torch.device('cuda:' + str(cuda_gpu))
        torch.backends.cudnn.benchmark = True
        # 使用cuDNN训练加速的标志，一般设置True即可自动对网络训练（卷积方式等）优化
    else:
        device = torch.device('cpu')

    if log_en:
        # 设置工作文件夹，存储训练可视化文件
        working_folder = os.path.join('experiments', str(time.strftime("%d.%m.%Y")),  str(time.strftime("%H.%M.%S")))
        if not os.path.exists(working_folder):
            os.makedirs(working_folder)

    x_tr, y_tr, y_tr_l, x_vl, y_vl, y_vl_l = data_pp_tools.load_data(args)
    # 加载解析原数据库
    model = cnn_lstm_model.model_builder(args)  # 初始化CNN-LSTM分类模型

    # batch化数据，数据预处理
    sq_len_tr = x_tr[0].shape[0]
    sq_len_vl = x_vl[0].shape[0]
    x_tr = torch.stack(x_tr, 0)      # [Nsamples, 40, 32, 32]
    y_tr = torch.stack(y_tr, 0)      # [Nsamples, 40]
    x_vl = torch.stack(x_vl, 0)      # [Nsamples_vld, 40, 32, 32]
    y_vl = torch.stack(y_vl, 0)      # [Nsamples_vld, 40]
    x_tr_ch = torch.split(x_tr, batch_size, dim=0)
    x_tr_ch = x_tr_ch[0:-1]
    y_tr_ch = torch.split(y_tr, batch_size, dim=0)
    y_tr_ch = y_tr_ch[0:-1]
    Ntrain = x_tr.shape[0]
    Nvld = x_vl.shape[0]

    cost_func = torch.nn.NLLLoss()
    adam_opt = torch.optim.Adam(lr=adam_lr, params=model.parameters())

    loss_arr = []
    loss_arr_vld = []
    acc_arr = []
    acc_arr_vld = []

    for i in range(0, epochs):
        # 开启一个epoch
        adam_opt.zero_grad()   # 每次进行新的epoch前清零积累的梯度
        for (x, y) in zip(x_tr_ch, y_tr_ch):
            y_logp_pred = model.forward(x.view(-1, 32, 32).unsqueeze(1), sq_len_tr, batch_size, False)
            # [input, seq_len, batch size, stateful flag]
            loss = cost_func(y_logp_pred[:, -1], y[:, -1])
            # 计算一个batch训练数据的损失
            loss.backward()
            adam_opt.step()
            adam_opt.zero_grad()

        # 一个epoch完成，计算整个训练数据的损失
        with torch.no_grad():
            y_logp_pred = model.forward(x_tr.view(-1, 32, 32).unsqueeze(1), sq_len_tr, Ntrain, False)
            # [input, seq_len, batch size, stateful flag]
            class_pred = torch.argmax(y_logp_pred[:, -1], dim=1).long()
            loss = cost_func(y_logp_pred[:, -1], y_tr[:, -1])

            acc_arr.append(1-tools.calc_error(class_pred, y_tr_l))
            loss_arr.append(loss)
            print('Training set epoch  : ', i, ' Avg NLL:    ', "{:10.5f}".format(loss_arr[-1]), '   Accuracy:    ',"{:10.6f}".format(acc_arr[-1]))

        # 一个epoch完成，计算整个验证数据的损失
        with torch.no_grad():
            y_logp_pred = model.forward(x_vl.view(-1, 32, 32).unsqueeze(1), sq_len_tr, Nvld, False)
            # [input, seq_len, batch size, stateful flag]
            class_pred = torch.argmax(y_logp_pred[:, -1], dim=1).long()
            loss = cost_func(y_logp_pred[:, -1], y_vl[:, -1])

            acc_arr_vld.append(1 - tools.calc_error(class_pred, y_vl_l))
            loss_arr_vld.append(loss)
            print('Validation set epoch: ', i, ' Avg NLL:    ', "{:10.5f}".format(loss_arr_vld[-1]), '   Accuracy:    ',"{:10.6f}".format(acc_arr_vld[-1]))

        # epoch完成，如果需要，对训练数据进行随机打乱，开始下一轮epoch
        if shuffle_data == 1:
            x_tr, y_tr, x_tr_ch, y_tr_ch, y_tr_l = tools.shuffle_data(x_tr, y_tr, y_tr_l, batch_size)

    if log_en:
        # 日志记录训练过程
        logging.basicConfig(filename=working_folder + '\\info.log', level=logging.INFO)
        logging.info('Model: %s', model)
        logging.info('Epochs: %d', epochs)
        logging.info('Batch size: %d', batch_size)
        logging.info('Pre training Accuracy %f', acc_arr[0])
        logging.info('Pre training NLL %f', loss_arr[0])
        logging.info('Post training Accuracy %f', acc_arr[-1])
        logging.info('Post training NLL %f', loss_arr[-1])
        # 模型参数保存
        torch.save(model.state_dict(), os.path.join(working_folder, 'NN_model.pkl'))
        torch.save(torch.stack(acc_arr), 'train_acc.pt')
        torch.save(torch.stack(loss_arr), 'train_nll.pt')
        torch.save(torch.stack(acc_arr_vld), 'validation_acc.pt')
        torch.save(torch.stack(loss_arr_vld), 'validation_nll.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments')

    # 添加辅助参数
    parser.add_argument('--cuda_gpu', help='defulat GPU number', default=0)
    parser.add_argument('--save_model', help='bool to indicate if to save NN model', default=1)
    parser.add_argument('--log_en', help='bool to save log of the script in the working folder', default=1)
    parser.add_argument('--plot_show', help='bool to indicate whether to show figures at the end of run', default=0)

    # 添加数据解析和预处理的参数
    parser.add_argument('--reshape_flag', help='If 1 then reshape data to 32x32 matrices', default=1)
    parser.add_argument('--zero_pad', help='If true, zero pad train & validation sequences to max length, if Int than pad/subsample to Int', default=40)
    parser.add_argument('--Nlabels', help='Number of used labels from the data set', default=11)

    # CNN-LSTM模型参数
    parser.add_argument('--Cin', help='Input channels', default=1)
    parser.add_argument('--Cout1', help=' Channels Conv Layer 1', default=32)
    parser.add_argument('--Cout2', help='Channels Conv layer 2', default=16)
    parser.add_argument('--Cout3', help='Channels Conv layer 3', default=8)
    parser.add_argument('--Lin_lstm', help='Input size to LSTM after FC', default=100)
    parser.add_argument('--lstm_hidden_size', help='LSTM hidden size', default=100)
    parser.add_argument('--lstm_num_layers', help='Number of LSTM layers', default=1)
    parser.add_argument('--batch_size', help='Default batch size', default=64)

    # 训练参数
    parser.add_argument('--epochs', help='Num of epochs for training', default=40)
    parser.add_argument('--adam_lr', help='Adam optimizer initial LR', default=1e-3)
    parser.add_argument('--shuffle_data', help='Flag to shuffle data after each epoch', default=1)

    args = vars(parser.parse_args())    # 解析添加的参数
    pp.pprint(args)  # pp.pprint() 美观输出
    main(args)   # 训练

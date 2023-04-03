import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=bool, required=False, default=True, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False,
                    help='perform training on full input dataset without validation and testing')
parser.add_argument('--test_only', type=bool, default=False,
                    help='perform test on full input dataset without training and validation')
parser.add_argument('--model_id', type=str, required=False, default='test-whatever', help='model id')
parser.add_argument('--model', type=str, required=False, default='Informer',
                    help='model name, options: [Autoformer, Informer, Transformer, DLinear, OurCNN]')
parser.add_argument('--pic_save_key', type=str, required=False, default=None,
                    help='By default, the images will be named train.jpg and test.jpg, respectively. '
                         'If you need to, you can specify the preceding keyword, like:[keys]_train.jpg')

# data loader
parser.add_argument('--data', type=str, required=False, default='ETTm1',
                    help='options:[ETTH1, ETTh2, ETTm1, ETTm2, Traffic, Electricity, Exchange—Rate, Weather, ILI, RC]')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTm1.csv',
                    help='options:[ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv, RC.xlsx, traffic.csv, weather.csv]')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, '
                         'S:univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers
# parser.add_argument('--pooling_method', type=str, default='max')
parser.add_argument('--embed_type', type=int, default=0,
                    help='0: default； This has the same effect as pattern 1'
                         '1: value embedding + temporal embedding + positional embedding '
                         '2: value embedding + temporal embedding '
                         '3: value embedding + positional embedding '
                         '4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', default=False, help='whether to predict unseen future data')
# CNN
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--layer_num', type=int, default=15)
parser.add_argument('--auto_capture', type=bool, default=False,
                    help='the enable is to control wheather auto capture the layer num')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=144, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=4, help='early stopping patience')  # 当连续patience个epoch的loss>历史最低则默认已经找到最佳模型
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')       # 0.0001
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', help='use automatic mixed precision training', default=False)

# criterion
parser.add_argument('--criterion', type=str, help='options[mse, mae, huber]', default='huber')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main
exp = Exp(args)  # set experiments

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        # setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        #     args.model_id,
        #     args.model,
        #     args.data,
        #     args.features,
        #     args.seq_len,
        #     args.label_len,
        #     args.pred_len,
        #     args.d_model,
        #     args.n_heads,
        #     args.e_layers,
        #     args.d_layers,
        #     args.d_ff,
        #     args.factor,
        #     args.embed,
        #     args.distil,
        #     args.des, ii)
        print('>>>>>>> start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train()

        if not args.train_only:
            print('>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test()

        if args.do_predict:
            print('>>>>>>> predicting <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.predict(True)

        torch.cuda.empty_cache()    # 在Nvidia-smi中释放显存
elif args.test_only:
    # path = f'./checkpoints/DLinear/id_test-m_DLinear-d_ETTm1-f_M-s_336-p_336-m_25-b_8-amp_1.pth'
    path = './checkpoints/{}/id_{}-m_{}-d_{}-f_{}-s_{}-p_{}-m_{}-b_{}-amp_{}.pth'.format(
        args.model, args.model_id, args.model, args.data, args.features,
        args.seq_len, args.pred_len, args.moving_avg, args.batch_size, 1 if args.use_amp else 0
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path '{path}' does not exist")
    exp.test(args, checkpoints_path=args.pth_path)

# else:
#     ii = 0
#     setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
#                                                                                                   args.model,
#                                                                                                   args.data,
#                                                                                                   args.features,
#                                                                                                   args.seq_len,
#                                                                                                   args.label_len,
#                                                                                                   args.pred_len,
#                                                                                                   args.d_model,
#                                                                                                   args.n_heads,
#                                                                                                   args.e_layers,
#                                                                                                   args.d_layers,
#                                                                                                   args.d_ff,
#                                                                                                   args.factor,
#                                                                                                   args.embed,
#                                                                                                   args.distil,
#                                                                                                   args.des, ii)
#
#     exp = Exp(args)  # set experiments
#
#     if args.do_predict:
#         print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#         exp.predict(setting, True)
#     else:
#         print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#         exp.test(setting, test=1)
#     torch.cuda.empty_cache()
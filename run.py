# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2021 THUML @ Tsinghua University
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Autoformer (https://arxiv.org/pdf/2106.13008.pdf) implementation
# from https://github.com/thuml/Autoformer by THUML @ Tsinghua University
####################################################################################

import argparse
import torch
from exp.exp_main import Exp_Main
from exp.exp_diffusion import Exp_Diffusion
from exp.exp_rnn import Exp_RNN
import random
import numpy as np


if __name__ == "__main__":
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, required=True, default='Transplit',
            help=('model name, options: [Transplit, Autoformer, Informer, Transformer, Reformer, FEDformer]'
                  'and their MS versions: [TransplitMS, AutoformerMS, InformerMS, etc]'))

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help=('forecasting task, options:[M, MS, S, SA]; '
                              'M:multivariate predict multivariate, '
                              'MS:multivariate predict univariate, '
                              'S:univariate predict univariate, '
                              'SA:univariate predict univariate (use all features)'))
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help=('freq for time features encoding, options:['
                              's:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly'
                              '], you can also use more detailed freq like 15min or 3h'))
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--save_preds', action='store_true', help='save predictions', default=False)

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length (Autoformer)')
    parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length')
    parser.add_argument('--period', type=int, default=24, help='interval between each sample')

    # supplementary config for EVT model
    parser.add_argument('--window_size', type=int, default=24, help='window size, for EVT')
    parser.add_argument('--num_windows', type=int, default=7, help='num of windows, for EVT')

    # supplementary config for FiLM model
    parser.add_argument('--modes1', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--mode_type',type=int,default=0)

    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Wavelets',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='low',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    # supplementary config for Reformer model
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--film_ours', default=True, action='store_true')
    parser.add_argument('--ab', type=int, default=2, help='ablation version')
    parser.add_argument('--ratio', type=float, default=0.5, help='dropout')
    parser.add_argument('--film_version', type=int, default=0, help='compression')

    # supplementary config for Diffusion models
    parser.add_argument('--num_steps', type=int, default=50, help='diffusion steps')
    parser.add_argument('--condition_prob', type=float, default=0.8, help='condition probability during training')
    parser.add_argument('--blur_sigma_min', type=float, default=0.5, help='Minimum blurring sigma')
    parser.add_argument('--blur_sigma_max', type=float, default=16.0, help='Maximum blurring sigma')
    parser.add_argument('--training_noise', type=float, default=0.01, help='noise factor for the forward process')
    parser.add_argument('--sampling_noise', type=float, default=0.00, help='noise factor for the backward process')
    parser.add_argument('--use_causal', action='store_true', default=False, help='use causal diffusion')
    parser.add_argument('--transformer_model', type=str, default='', help='transformer model to use for the base forecast')
    parser.add_argument('--transformer_checkpoint', type=str, default='', help='.pth file to load for the base forecast')
    parser.add_argument('--transformer_blurring', type=int, default=7, help='num of blurring steps to apply to the transformer\'s output')
    parser.add_argument('--transformer_deblurring', type=int, default=15, help='num of deblurring (backward) steps')

    # model hyperparameters
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=5, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

    args = parser.parse_args()

    if 'MS' in args.model:
        args.use_multi_scale = True

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        args.devices = '0'
        for i in range(1, torch.cuda.device_count()):
            args.devices = args.devices + f',{i}'
        args.use_multi_gpu = True if torch.cuda.device_count()>1 else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    if args.data_path == 'weather.csv':
        args.root_path = './dataset/weather/' 
        c = 21
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
    elif args.data_path == 'traffic.csv':
        args.root_path = './dataset/traffic/' 
        c = 862
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
    elif args.data_path == 'electricity.csv':
        args.root_path = './dataset/electricity/' 
        c = 321
        args.enc_in = c
        args.dec_in = c
        args.c_out = c
    if args.features == 'SA':
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1

    print('Args in experiment:')
    print(args)

    if args.model in ['InverseHeatDissipation']:
        Exp = Exp_Diffusion
        args.features = 'SA'
        # args.enc_in = 1
        # args.dec_in = 1
        # args.c_out = 1
    elif args.model in ['EVT']:
        Exp = Exp_RNN
        args.features = 'SA'
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.period = args.pred_len
    else:
        Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = f'{args.data_path[:-4]}_{args.model}_{args.pred_len}_{args.loss}_{ii}'
            exp = Exp(args)
            print('>>>>>>> start training: {} >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>> testing: {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>> predicting: {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        for ii in range(args.itr):
            setting = f'{args.data_path[:-4]}_{args.model}_{args.pred_len}_{args.loss}_{ii}'

            if args.model in ['InverseHeatDissipation']:
                args.features = 'M'
                args.transformer_setting = f"{args.data_path[:-4]}_{args.transformer_model}_{args.pred_len}_{args.loss}_{ii}"
                args.transformer_checkpoint = f"./{args.checkpoints}/{args.transformer_setting}/checkpoint.pth"
            exp = Exp(args)
            print('>>>>>>> testing: {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, load=True)
            torch.cuda.empty_cache()

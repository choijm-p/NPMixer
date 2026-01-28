import argparse
import os
import random
import numpy as np
import torch
from exp.exp_main import Exp_Main
import optuna
import logging
import sys
import uuid
from datetime import datetime
import traceback
import shutil
from torchsummary import summary
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def int_list(s):
    if isinstance(s, list):
        return s
    return list(map(int, s.split(',')))

def parse_patching(patch_str):
    try:
        patch_len_str, stride_str, padding = patch_str.split(',')
        patch_len = int(patch_len_str)
        stride = int(stride_str)
        return {'patch_len': patch_len, 'stride': stride, 'padding': padding}
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid patching configuration: '{patch_str}'. Expected format 'patch_len,stride,padding'.")

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# Basic Config
parser.add_argument('--task_name', type=str, default='long_term_forecast')
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--train_only', type=bool, default=False)
parser.add_argument('--model_id', type=str, default='test')

# Optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# Data Loader
parser.add_argument('--model', type=str, default='NPMixer')
parser.add_argument('--data', type=str, default='ETTh1')
parser.add_argument('--root_path', type=str, default='./dataset/')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--fix_seed', type=int, default=3000)

# Forecasting Dimensions
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--enc_in', type=int, default=7)             

# NPMixer
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--e_layers', type=int, default=5, help='num of encoder layers (Channel-Mixing)')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--dropout', type=float, default=0.5381, help='dropout probability')
parser.add_argument('--activation', type=str, default='gelu', help='activation function')
parser.add_argument('--wavelet', type=str, default='db1', help='initial wavelet basis (e.g., db1, haar)')
parser.add_argument('--wavelet_j', type=int, default=3, help='number of SWT decomposition levels (J)')
parser.add_argument('--trainable_wavelet', type=str2bool, default=True, help='enable Learnable SWT filters')
parser.add_argument('--patch_len', type=int, default=16, help='patch length (P)')
parser.add_argument('--stride', type=int, default=8, help='stride between patches')
parser.add_argument('--max_mix_levels', type=int, default=3, help='number of hierarchical mixing stages (K)')
parser.add_argument('--alpha', type=float, default=0.0, help='initial value for learnable gate alpha')
parser.add_argument('--output_attention', type=str2bool, default=False, help='whether to output attention in encoder')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--fc_dropout', type=float, default=0.1, help='fully connected dropout')

# RevIN
parser.add_argument('--revin', type=str2bool, default=True)
parser.add_argument('--affine', type=str2bool, default=False)

# Device
parser.add_argument("--use_gpu", type=str2bool, nargs='?',const=True, default=False,help="use gpu")
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')


args = parser.parse_args()
fix_seed = args.fix_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.autograd.set_detect_anomaly(True)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and getattr(args, "use_multi_gpu", False):
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):

        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_df{}_des{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.e_layers,
            args.d_ff,
            args.des,
            ii
        )

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        _, vali_loss = exp.train(setting)

        if not args.train_only:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

        if getattr(args, "do_predict", False):
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()

else:
    ii = 0

    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_df{}_des{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.e_layers,
        args.d_ff,
        args.des,
        ii
    )

    exp = Exp(args)

    if getattr(args, "do_predict", False):
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)

    torch.cuda.empty_cache()

#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from data_process import sign_gendata


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Decoupling Graph Convolution Network with DropGraph Module')
    
    parser.add_argument(
        '--config',
        default='../config/joint.yaml',
        help='path to the configuration file')

    parser.add_argument(
        '--data-path',
        default= None,
        help='path to the input data')

    # model
    parser.add_argument('--model',
     default= None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default= None,
        help='the weights for model')

    parser.add_argument('--label',
        type=dict,
        default= dict(), help='label mapping')
    return parser

def print_time():
    localtime = time.asctime(time.localtime(time.time()))
    print_log("Local current time :  " + localtime)

def print_log(str, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + ' ] ' + str
    print(str)
    # if self.arg.print_log:
    #     with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
    #         print(str, file=f)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)  
        # mod 객체에 string으로 함수, 클래스 접근하기위함
    return mod



def load_model(arg):
    Model = import_class(arg.model)
    model = Model(**arg.model_args).cuda(0)
    weights = torch.load(arg.weights)
    weights = OrderedDict(
            [[k.split('module.')[-1], v.cuda(0)] for k, v in weights.items()]) 
    
    model.load_state_dict(weights)

    return model

def inference(arg, model):
    #print_log('inference start')
    model.eval()
    with torch.no_grad():
        data = np.load(arg.data_path, mmap_mode='r')
        data_numpy = np.array(data)
        # normalization
        data_numpy[0,0,:,:,:] = data_numpy[0,0,:,:,:] - data_numpy[0,0,:,0,0].mean(axis=0)
        data_numpy[0,1,:,:,:] = data_numpy[0,1,:,:,:] - data_numpy[0,1,:,0,0].mean(axis=0)
        
        data_numpy = torch.tensor(data_numpy)
        data = Variable(data_numpy.float().cuda(0),requires_grad=False)
        output = model(data)
        print('output : ', output)
        _, predict_label = torch.max(output.data, 1)
        #predict = list(predict_label.cpu().numpy())
        predict = predict_label.cpu().numpy()[0]
        #print_log('predict : {}'.format(predict))
        #print_log('inference end')
        return arg.label[predict]


def pipeline(data):
    sign_gendata.preprocess(data)

    parser = get_parser()
    p = parser.parse_args()
    
    print(os.getcwd())
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)  # 위 get_parser의 default를 yaml data로 채워넣는거

    arg = parser.parse_args()
    init_seed(0)
    #print(' arg : ', arg)
    model = load_model(arg)
    return inference(arg,model)


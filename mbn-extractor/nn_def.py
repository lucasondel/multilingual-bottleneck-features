#!/usr/bin/env python

########################################################################################
#  copyright (C) 2017 by Anna Silnova, Pavel Matejka, Oldrich Plchot, Frantisek Grezl  #
#                         Brno Universioty of Technology                               #
#                         Faculty of information technology                            #
#                         Department of Computer Graphics and Multimedia               #
#  email             : {isilnova,matejkap,iplchot,grezl}@vut.cz                        #
########################################################################################
#                                                                                      #
#  This software and provided models can be used freely for research                   #
#  and educational purposes. For any other use, please contact BUT                     #
#  and / or LDC representatives.                                                       #
#                                                                                      #
########################################################################################

import utils
import numpy as np

dct_basis = 6
left_ctx=right_ctx=5
dct_xform =  utils.dct_basis(dct_basis, left_ctx+right_ctx+1)
dct_xform[0] = np.sqrt(2./(left_ctx+right_ctx+1))
hamming_dct = (dct_xform*np.hamming(left_ctx+right_ctx+1)).T

def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))

def softmax_fun(x):
    a=np.exp(x).T
    return (a/np.sum(a,axis=0).T).T

def preprocess_nn_input(X,left_ctx=left_ctx,right_ctx=right_ctx):
    X = utils.framing(X, left_ctx+1+right_ctx).transpose(0,2,1)
    return np.dot(X.reshape(-1,hamming_dct.shape[0]), hamming_dct).reshape(X.shape[0], -1)


def create_nn_extract_st_BN(X, param_dict, bn_position=2):
    mean = param_dict['input_mean']
    std  = param_dict['input_std']
    #instead of mean and standard deviation we store negative mean and inverse std
    Y = (X + mean) * std
    num_of_layers=(len(param_dict.keys())-5) // 2
    # n_hidden_before_BN --> sigmoid
    # BN activation --> linear
    for ii, f in enumerate([lambda x: sigmoid_fun(x)]*bn_position+[lambda x:x]):
        W = param_dict['W'+str(ii+1)]
        b = param_dict['b'+str(ii+1)]
        Y = f(Y.dot(W) + b)
    Y1=np.hstack([Y[0:-20],Y[5:-15],Y[10:-10],Y[15:-5],Y[20:]])
    bn_mean=param_dict['bn_mean']
    bn_std=param_dict['bn_std']
    Y1=(Y1+bn_mean)*bn_std
    for ii, f in enumerate([lambda x: sigmoid_fun(x)]*(num_of_layers-bn_position-2)+[lambda x:x]):
        W = param_dict['W'+str(ii+bn_position+3)]
        b = param_dict['b'+str(ii+bn_position+3)]
        Y1 = f(Y1.dot(W) + b)
    return  Y1,Y


def create_nn_extract_posterior(Y, param_dict):
    num_of_layers=(len(param_dict.keys()))/2
    for ii, f in enumerate([lambda x: sigmoid_fun(x)]*(num_of_layers-1)+[lambda x : softmax_fun(x)]):
        W = param_dict['W'+str(ii+1)]
        b = param_dict['b'+str(ii+1)]
        Y = f(Y.dot(W) + b)
    return  Y

def create_nn_extract_posterior_ml(Y, param_dict):
    num_of_classes_per_lan=param_dict['num_cl']
    num_of_layers=(len(param_dict.keys())-1)/2
    for ii, f in enumerate([lambda x: sigmoid_fun(x)]*(num_of_layers-1)):
        W = param_dict['W'+str(ii+1)]
        b = param_dict['b'+str(ii+1)]
        Y = f(Y.dot(W) + b)
    W = param_dict['W'+str(ii+2)]
    b = param_dict['b'+str(ii+2)]
    Y=Y.dot(W) + b
    lan_start_ind=0
    for num_cl in num_of_classes_per_lan:
        num_cl=int(num_cl)
        Y[:,lan_start_ind:lan_start_ind+num_cl]=softmax_fun(Y[:,lan_start_ind:lan_start_ind+num_cl])
        lan_start_ind+=num_cl
    return  Y


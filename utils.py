from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import copy
import os
import scipy.io as scio

import numpy as np
import random
random.seed(0)



'''
Tools
'''
def coefficient(category_1, category_2, sample1_label, sample2_label):
    cls_bool1 = (sample1_label == category_1)
    cls_bool2 = (sample2_label == category_2)
    # print('cls_bool1 size:', cls_bool1.size())
    # print('cls_bool2 size:', cls_bool2.size())

    # cls_bool1 = cls_bool1.unsqueeze(1)
    # cls_bool2 = cls_bool2.unsqueeze(1)

    total_cls = torch.cat([cls_bool1.int(), cls_bool2.int()], dim=0)

    total_coef = torch.ger(total_cls.view(-1).cpu(), total_cls.view(-1).cpu())
    return total_coef
def lsd(source, target, source_label, target_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    device = source.device  # 获取输入张量的设备信息
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma).to(device)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    intra_lsd_val = []
    inter_lsd_val = []
    num_class = 2
    for c1 in range(num_class):
        for c2 in range(num_class):
            coef_val = coefficient(c1, c2, source_label, target_label).to(device)
            e_ss = torch.div(coef_val[:n, :n] * XX, (coef_val[:n, :n]).sum() + 1e-5)
            e_st = torch.div(coef_val[:n, n:] * XY, (coef_val[:n, n:]).sum() + 1e-5)
            e_ts = torch.div(coef_val[n:, :n] * YX, (coef_val[n:, :n]).sum() + 1e-5)
            e_tt = torch.div(coef_val[n:, n:] * YY, (coef_val[n:, n:]).sum() + 1e-5)
            # e_ss[e_ss != e_ss] = 0
            # e_st[e_st != e_st] = 0
            # e_ts[e_ts != e_ts] = 0
            # e_tt[e_tt != e_tt] = 0

            lsd_val = e_ss.sum() + e_tt.sum() - e_st.sum() - e_ts.sum()
            """if lsd_val.is_nan():
                    continue
            else:"""
            if c1 == c2:
                intra_lsd_val.append(lsd_val)
            elif c1 != c2:
                inter_lsd_val.append(lsd_val)

    loss = sum(intra_lsd_val) / len(intra_lsd_val) - sum(inter_lsd_val) / len(inter_lsd_val)
    return loss
def norminx(data):
    '''
    description: norm in x dimension
    param {type}:
        data: array
    return {type} 
    '''
    for i in range(data.shape[0]):
        data[i] = normalization(data[i])
    return data


def norminy(data):
    dataT = data.T
    for i in range(dataT.shape[0]):
        dataT[i] = normalization(dataT[i])
    return dataT.T


def normalization(data):
    '''
    description: 
    param {type} 
    return {type} 
    '''
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# package the data and label into one class


class CustomDataset(Dataset):
    # initialization: data and label
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    # get the size of data

    def __len__(self):
        return len(self.Data)
    # get the data and label

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        return data, label

# mmd loss and guassian kernel


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1) % batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)

    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4*d*4)
    return loss


def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def PADA(features, ad_net, grl_layer, weight_ad, use_gpu=True):
    ad_out = ad_net(grl_layer(features))
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(
        np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    if use_gpu:
        dc_target = dc_target.cuda()
        weight_ad = weight_ad.cuda()
    return torch.nn.BCELoss(weight=weight_ad.view(-1))(ad_out.view(-1), dc_target.view(-1))



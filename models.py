
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import utils

from torch.nn import Parameter

import torch.nn.functional as F

class FMLayer1(torch.nn.Module):  ##3维
    def __init__(self, n, k, o):
        super(FMLayer1, self).__init__()
        self.n = n
        self.k = k
        self.o = o
        self.lin = torch.nn.Linear(self.n, self.o)  # 前两项线性层
        self.V = torch.nn.Parameter(torch.zeros(self.o, self.n, self.k))  # 交互矩阵
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)
        torch.nn.init.xavier_uniform_(self.V)  #

    def fm_layer(self, x):
        linear_part0 = self.lin(x)
        interaction_part_10 = torch.matmul(x, self.V)
        interaction_part_10 = torch.pow(interaction_part_10, 2)
        interaction_part_20 = torch.matmul(torch.pow(x, 2), torch.pow(self.V, 2))
        output0 = linear_part0 + 0.5 * torch.sum(interaction_part_20 - interaction_part_10, 2, keepdim=False).transpose(
            0, 1)  # .unsqueeze(2)
        return output0

    def forward(self, x):
        return self.fm_layer(x)


class FMLayer0(torch.nn.Module):  ####2维重复
    def __init__(self, n, k, o):
        super(FMLayer0, self).__init__()
        self.n = n
        self.k = k
        self.o = o
        self.linear0 = torch.nn.Linear(self.n, self.o)  # 前两项线性层
        self.V = torch.nn.Parameter(torch.randn(self.n, self.k))  # 交互矩阵

    def fm_layer(self, x):
        linear_part = self.linear0(x)
        interaction_part_1 = torch.matmul(x, self.V)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.matmul(torch.pow(x, 2), torch.pow(self.V, 2))
        p = 0.5 * torch.sum(interaction_part_1 - interaction_part_2, -1, keepdim=True)
        rp = p.repeat(1, self.o)
        output = linear_part + rp
        return output

    def forward(self, x):
        return self.fm_layer(x)

    # 基于tensorflow转换为torch


class FMLayer(nn.Module):
    def __init__(self, output_dim, factor_order, activation='softmax'):
        super(FMLayer, self).__init__()

        self.output_dim = output_dim
        self.factor_order = factor_order
        self.activation = nn.Softmax(dim=1) if activation == 'softmax' else None
        self.input_dim = None

        self.w = nn.Parameter(torch.Tensor(output_dim, factor_order))
        self.v = nn.Parameter(torch.Tensor(self.input_dim, factor_order))
        # 在第26行后面添加如下一行代码
        fm_layer = FMLayer((32, 50, 32))  # 将元组(32, 50)作为参数传递给FMLayer类的构造函数
        self.b = nn.Parameter(torch.Tensor(output_dim))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.v)
        nn.init.zeros_(self.b)

    def forward(self, inputs):
        if self.input_dim is None:
            self.input_dim = inputs.shape[1]
            self.v = nn.Parameter(torch.Tensor(self.factor_order, self.input_dim))
            nn.init.xavier_uniform_(self.v)

        X_square = torch.square(inputs)

        xv = torch.square(torch.mm(inputs, self.v))
        xw = torch.mm(inputs, self.w)

        p = 0.5 * torch.sum(xv - torch.mm(X_square, torch.square(self.v)), dim=1)
        rp = torch.repeat_interleave(p.view(-1, 1), repeats=self.output_dim, dim=1)

        f = xw + rp + self.b

        output = f.view(-1, self.output_dim)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def extra_repr(self):
        return f"output_dim={self.output_dim}, factor_order={self.factor_order}"


# 定义标签平滑的交叉熵损失函数
def label_smoothed_nll_loss(log_probs, target, eps, ignore_index=-100):
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1))
    nll_loss = nll_loss.masked_fill(target.eq(ignore_index), 0)
    
    smooth_loss = -log_probs.mean(dim=-1)
    
    loss = (1.0 - eps) * nll_loss + eps * smooth_loss
    #return loss.sum()
    return loss.mean()  # 用.mean()替代.sum()，确保在不同batch大小时结果可比较

class GaussianNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.0).cuda()

    def forward(self, x):
        if self.training:
            sampled_noise = self.noise.repeat(*x.size()).normal_(mean=0, std=self.sigma)
            x = x + sampled_noise
        return x

class CFE(nn.Module):
    def __init__(self):
        super(CFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(50, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #为了解决ReLU的死神经元问题，LeakyReLU被提出。它允许小的负梯度当输入值小于0
            #nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            #GaussianNoise(1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),#自己又添加的
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            #nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            #GaussianNoise(1.0),
        )

    def forward(self, x):
        x = self.module(x)
        return x


def pretrained_CFE(pretrained=False):
    model = CFE()
    if pretrained:
        pass
    return model


class pre_trained_MLP(nn.Module):
    def __init__(self):
        super(pre_trained_MLP, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(50, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(64,2)
        )

    def forward(self, x):
        x = self.module(x)
        return x


class DSFE(nn.Module):
    def __init__(self):
        super(DSFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(64, 32),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm1d(32, eps=1e-05, momentum=0.1,
                 #affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        # self.dropout = nn.Dropout(0.2)
        # self.flatten = nn.Flatten()
        self.normalize = nn.BatchNorm1d(32, eps=1e-05, momentum=0.1,
                                        affine=True, track_running_stats=True)
        self.fmlayer = FMLayer1(32, 100, 32)
        # self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.output = nn.Linear(32, 2)

    def forward(self, x):
        x = self.module(x)
        # x = self.dropout(x)
        # x = self.flatten(x)
        x = self.normalize(x)
        #x = self.fmlayer(x)
        #x = self.normalize(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.output(x)
        return x



class MSMDAERNet(nn.Module):
    def __init__(self, pretrained=False, number_of_source=15, number_of_category=4):
        super(MSMDAERNet, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # for i in range(1, number_of_source):
        #     exec('self.DSFE' + str(i) + '=DSFE()')
        #     exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=SLR_layer(32,' + str(number_of_category) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        cls_loss = 0
        mmd_loss = 0
        lsd_loss = 0
        transfer_loss = 0
        disc_loss = 0
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            # Each domian specific feature extractor
            # to extract the domain specific feature of target data
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_tgt_DSFE_i)
            # Use the specific feature extractor
            # to extract the source data, and calculate the mmd loss
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            # mmd_loss += utils.mmd(data_src_DSFE, data_tgt_DSFE[mark])
            mmd_loss += utils.mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])
            # discrepency loss
            for i in range(len(data_tgt_DSFE)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) -
                        F.softmax(data_tgt_DSFE[i], dim=1)
                    ))
            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            #cls_loss += F.nll_loss(F.log_softmax(
                #pred_src, dim=1), label_src.squeeze())
            pred_tgt = eval(DSC_name)(data_tgt_DSFE[mark])  #
            tar_softmax_output = nn.functional.softmax(pred_tgt, dim=1)
            max_prob, pseudo_label = torch.max(tar_softmax_output, dim=1)
            confident_bool = max_prob >= 0.60
            confident_example = data_tgt_DSFE[mark][confident_bool]  #
            confident_label = pseudo_label[confident_bool]

            if label_src.shape != confident_label.shape:
                if label_src.dim() == 1:
                    label_src = label_src.unsqueeze(1)
                elif confident_label.dim() == 1:
                    confident_label = confident_label.unsqueeze(1)

            lsd_loss += utils.lsd(data_src_DSFE, confident_example, label_src, confident_label)

            # 添加FBNM损失
            list_svd, _ = torch.sort(torch.sqrt(torch.sum(torch.pow(tar_softmax_output, 2), dim=0)), descending=True)
            transfer_loss += - torch.mean(list_svd[:min(tar_softmax_output.shape[0], tar_softmax_output.shape[1])])

            # 选择一个适当的平滑因子 eps，通常在 0.1 到 0.2 之间
            eps = 0.06
            # 将原有的交叉熵损失替换为标签平滑的交叉熵损失
            cls_loss = label_smoothed_nll_loss(F.log_softmax(pred_src, dim=1), label_src.squeeze(), eps)

            return cls_loss, mmd_loss, disc_loss , transfer_loss

        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))

            return pred






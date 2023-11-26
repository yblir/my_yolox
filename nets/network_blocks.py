# -*- encoding: utf-8 -*-

import torch
from torch import nn


class SiLU(nn.Module):
    '''
    silu激活函数写成类,是为了可以以类的形式调用
    '''

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name='silu', inplace=True):
    '''
    选取要使用的激活函数,一般使用silu,其他的都是陪衬
    :param name:
    :param inplace: 这个参数干嘛的?
    :return:
    '''
    if name == 'silu':
        loss_func = SiLU()
    elif name == 'relu':
        loss_func = nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        loss_func = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError(f'Unsuppport act type:{name}')

    return loss_func


# 不使用函数形式
def conv_bn_act(ch_in, ch_out, k_size, stride, groups=1, bias=False, act_name='silu'):
    '''
    卷积+批归一化+激活函数,YOLOx用的都是silu
    :param ch_in:输入通道数
    :param ch_out:输出通道数
    :param k_size:卷积核大小,新版pytorch建议传入元祖
    :param stride:卷积核每次移动的步长,也建议是元祖
    :param groups:分组数,当groups=ch_in时,是可分离卷积
    :param bias:偏置
    :param act_name:激活函数
    :return:
    '''
    pad = (k_size - 1) // 2  # 这步操作什么意思?
    return nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=k_size,
                                   stride=stride, padding=pad, groups=groups, bias=bias),
                         nn.BatchNorm2d(ch_out, eps=0.001, momentum=0.03),
                         get_activation(act_name, inplace=True))


class DwConvBnAct(nn.Module):
    '''
    写成类的形式,可以像nn.Conv一样作为模型的一层, 批归一化和激活函数计算后都合并到这一层了. summary模型时会更简洁
    '''

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = ConvBnAct(in_channels, in_channels, k_size=ksize, stride=stride, groups=in_channels, act_name=act)
        self.pconv = ConvBnAct(in_channels, out_channels, k_size=1, stride=1, groups=1, act_name=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

    # 推理时使用,融合了bn层
    def forward_fuse(self, x):
        return self.act(self.conv(x))


# 卷积+bn+silu 要写成ConvBnAct, 这样才能使用已经训练好的权重,很是郁闷
class ConvBnAct(nn.Module):
    '''
    写成类的形式,可以像nn.Conv一样作为模型的一层, 批归一化和激活函数计算后都合并到这一层了. summary模型时会更简洁
    '''

    def __init__(self, ch_in, ch_out, k_size, stride, groups=1, bias=False, act_name="silu"):
        '''
        卷积+批归一化+激活函数,YOLOx用的都是silu
        :param ch_in:输入通道数
        :param ch_out:输出通道数
        :param k_size:卷积核大小,新版pytorch建议传入元祖
        :param stride:卷积核每次移动的步长,也建议是元祖
        :param groups:分组数,当groups=ch_in时,是可分离卷积
        :param bias:偏置
        :param act_name:激活函数
        :return:
        '''
        super().__init__()
        pad = (k_size - 1) // 2
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(ch_out, eps=0.001, momentum=0.03)
        self.act = get_activation(act_name, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    # 推理时使用,融合了bn层
    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    '''
    这个模块用于隔行采样,w,h减半,通道变为原来4倍
    '''

    def __init__(self, ch_in, ch_out, k_size, stride, act_name='silu'):
        super(Focus, self).__init__()
        self.conv = ConvBnAct(4 * ch_in, ch_out, k_size=k_size, stride=stride, act_name=act_name)

    def forward(self, x):
        top_left = x[..., ::2, ::2]
        bot_left = x[..., 1::2, ::2]
        top_right = x[..., ::2, 1::2]
        bot_right = x[..., 1::2, 1::2]
        x = torch.cat([top_left, bot_left, top_right, bot_right], dim=1)

        return self.conv(x)


class SPPBottleneck(nn.Module):
    '''
    通过不同池化核大小的最大池化进行特征提取,提高网络的感受野
    '''

    def __init__(self, ch_in, ch_out, k_size=(5, 9, 13), activation='silu'):
        super(SPPBottleneck, self).__init__()
        hidden_ch1 = ch_in // 2  # 输出通道变为原来一半

        self.conv1 = ConvBnAct(ch_in, hidden_ch1, k_size=1, stride=1, act_name=activation)
        # 构建3个不同池化层的微模型
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in k_size])
        # 最后的输出还要经过卷积处理,这层的输入通道数是池化后的通道数+1个未处理的通道数
        hidden_ch2 = hidden_ch1 * (len(k_size) + 1)
        self.conv2 = ConvBnAct(hidden_ch2, ch_out, k_size=1, stride=1, act_name=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)  # 0维是batch层,1维是通道维度
        return self.conv2(x)


class Bottleneck(nn.Module):
    '''
    构建cspnet小的残差层
    '''

    def __init__(self, ch_in, ch_out, shortcut=True, expansion=0.5, dw=False, activation="silu"):
        '''
        :param ch_in:
        :param ch_out:
        :param shortcut: 是否添加跳线
        :param expansion: 通道扩张倍数
        :param dw: 是否使用深度可分离卷积
        :param activation:
        '''
        super(Bottleneck, self).__init__()
        hidden_ch = int(ch_out * expansion)  # 中间层的输出通道数
        Conv = DwConvBnAct if dw else ConvBnAct

        # 利用1x1卷积进行通道数的缩减。缩减率一般是50%, 减少参数量
        self.conv1 = ConvBnAct(ch_in, hidden_ch, k_size=1, stride=1, act_name=activation)
        # 利用3x3卷积进行通道数扩张,扩张至原来的应该的输出通道数. 并且完成特征提取
        self.conv2 = Conv(hidden_ch, ch_out, k_size=3, stride=1, act_name=activation)
        self.use_add = shortcut and ch_in == ch_out

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:  # 构建残差网络
            y = y + x
        return y


class CSPLayer(nn.Module):
    '''
    构建csp模块
    '''

    def __init__(self, ch_in, ch_out, repeat=1, shortout=True, expansion=0.5, dw=False, activation='silu'):
        super(CSPLayer, self).__init__()
        hidden_ch = int(ch_out * expansion)  # 中间层的输出通道数
        # 主干部分的初次卷积
        self.part1_conv = ConvBnAct(ch_in, hidden_ch, k_size=1, stride=1, act_name=activation)
        # 大的残差变的初次卷积
        self.part2_conv = ConvBnAct(ch_in, hidden_ch, k_size=1, stride=1, act_name=activation)
        # 对堆叠的结果进行卷积的处理,2*hidden_ch,是因为将要处理的输入的通道已经经过通道数合并
        # todo 之前写成merge_conv 不行,因为构建模型时键会包含merge_conv,就加载不了训练好的权重阿里
        # todo 另外, 模型中权重的排序次序是__init__中定义层的次序,不是forward中调用次序.因此若定义层次序不同,也加载不了训练好的权重
        self.merge_conv = ConvBnAct(2 * hidden_ch, ch_out, k_size=1, stride=1, act_name=activation)
        # 根据重复次数构建残差结构
        self.m = nn.Sequential(*[Bottleneck(hidden_ch, hidden_ch, shortcut=shortout,
                                            expansion=1.0, dw=dw, activation=activation) for _ in range(repeat)])

    def forward(self, x):
        # x1是主干部分
        x_1 = self.part1_conv(x)
        # x2是大的残差边部分
        x_2 = self.part2_conv(x)
        # 第二部分利用残差结构堆叠继续进行特征提取
        '''
        破案~, 之前使用x2 = self.m(x_2), 因为x_1,x_2都是从x一样的卷积得来,从0开始训练无所谓,但对于预加载模型,这样破坏了
        模型原来结构,导致训练权重不能工作.
        '''
        x_1 = self.m(x_1)
        # 经过多层残差处理的特征通道在前,与原始的YOLOx保持一直
        x = torch.cat((x_1, x_2), dim=1)

        return self.merge_conv(x)
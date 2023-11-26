import torch
from torch import nn

# from nets.darknet53 import ConvBnAct, CSPLayer, CSPDarkNet, DwConvBnAct
from nets.network_blocks import ConvBnAct, DwConvBnAct, Focus, SPPBottleneck, CSPLayer


class CSPDarkNet(nn.Module):
    '''
    构建cspdarkent主干特征提取网络
    '''

    def __init__(self, dep_mul, wid_mul, out_features=('dark3', 'dark4', 'dark5'), dw=False, activation='silu'):
        '''
        :param dep_mul: 用于计算csp模块中,小的残差重复次数
        :param wid_mul: 用于计算初始时的通道数量, 和上面的参数,推测是用于构建不同大小权重的多个模型
        :param out_features: 最后提取的输出特征层
        :param dw: 是否使用可分离卷积
        :param activation: 激活函数
        '''
        super(CSPDarkNet, self).__init__()

        base_ch = int(wid_mul * 64)  # 初始通道数是64
        base_depth = max(round(dep_mul * 3), 1)  # round用于向下取整
        Conv = DwConvBnAct if dw else ConvBnAct

        self.out_features = out_features
        # 3,640,640->12,320,320->base_ch(64),320,320
        self.stem = Focus(ch_in=3, ch_out=base_ch, k_size=3, stride=1, act_name=activation)

        # 完成卷积之后，320, 320, 64 -> 160, 160, 128
        # 完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        self.dark2 = nn.Sequential(Conv(base_ch, 2 * base_ch, k_size=3, stride=2, act_name=activation),
                                   CSPLayer(2 * base_ch, 2 * base_ch, repeat=base_depth, dw=dw, activation=activation))

        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        self.dark3 = nn.Sequential(Conv(2 * base_ch, 4 * base_ch, k_size=3, stride=2, act_name=activation),
                                   CSPLayer(4 * base_ch, 4 * base_ch,
                                            repeat=base_depth * 3, dw=dw, activation=activation))

        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        self.dark4 = nn.Sequential(Conv(4 * base_ch, 8 * base_ch, k_size=3, stride=2, act_name=activation),
                                   CSPLayer(8 * base_ch, 8 * base_ch,
                                            repeat=base_depth * 3, dw=dw, activation=activation))

        #   完成卷积之后，512,40, 40,  -> 1024,20, 20,
        #   完成SPP之后，1024,20, 20 -> 1024,20, 20,
        #   完成CSPlayer之后，1024,20, 20 -> 1024,20, 20
        self.dark5 = nn.Sequential(Conv(8 * base_ch, 16 * base_ch, k_size=3, stride=2, act_name=activation),
                                   SPPBottleneck(16 * base_ch, 16 * base_ch, k_size=(5, 9, 13)),
                                   CSPLayer(16 * base_ch, 16 * base_ch,
                                            repeat=base_depth, shortout=False, dw=dw, activation=activation))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x
        x = self.dark2(x)
        outputs['dark2'] = x

        # dark3的输出为256,80, 80，是一个有效特征层
        x = self.dark3(x)
        outputs['dark3'] = x

        # dark4的输出为512,40, 40，是一个有效特征层
        x = self.dark4(x)
        outputs['dark4'] = x

        # dark5的输出为1024,20, 20，是一个有效特征层
        x = self.dark5(x)
        outputs['dark5'] = x

        # 构建输出层字典
        return {k: v for k, v in outputs.items() if k in self.out_features}


# YOLOPAFPN
class PANet(nn.Module):
    '''
    路径聚合网络,加强特征提取网络
    '''

    def __init__(self, depth=1.0, width=1.0, in_feats=('dark3', 'dark4', 'dark5'),
                 ch_in=(256, 512, 1024), dw=False, activation='silu'):
        '''
        :param depth:用于csp模块中小残差块的重复次数
        :param width:用于计算初始时的通道数量, 和上面的参数,推测是用于构建不同大小权重的多个模型
        :param in_feats:主干网络待提取的三个特征层
        :param ch_in:
        :param dw:
        :param activation:
        '''
        super(PANet, self).__init__()
        Conv = DwConvBnAct if dw else ConvBnAct

        self.backbone = CSPDarkNet(dep_mul=depth, wid_mul=width, dw=dw, activation=activation)
        self.in_feats = in_feats
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # todo 这些属性的名称也必须与原始模型名称一致,否则不能使用权重,
        #  当使用model_train.load_state_dict(torch.load(path,strict=False))时不再报错,但会跳过键名不一样的权重,不加载.
        # 主干特征提取网络后经过卷积,可得panet第一个输入,也是fpn段的第一次个输入. 20, 20, 1024 -> 20, 20, 512
        self.last_conv = ConvBnAct(int(width * ch_in[2]), int(width * ch_in[1]), k_size=1, stride=1,
                                   act_name=activation)
        # 40, 40, 1024 -> 40, 40, 512
        self.P4_P3_csp = CSPLayer(int(2 * width * ch_in[1]), int(width * ch_in[1]),
                                  repeat=round(3 * depth), shortout=False, dw=dw, activation=activation)
        # 40, 40, 512 -> 40, 40, 256
        self.P4_conv = ConvBnAct(int(width * ch_in[1]), int(width * ch_in[0]), k_size=1, stride=1,
                                 act_name=activation)
        self.P3_csp = CSPLayer(int(2 * width * ch_in[0]), int(width * ch_in[0]),
                               repeat=round(3 * depth), shortout=False, dw=dw, activation=activation)
        # 下采样阶段,使用卷积完成下采样过程. 有可能会用可分离卷积?
        self.down_conv1 = Conv(int(width * ch_in[0]), int(width * ch_in[0]), k_size=3, stride=2, act_name=activation)
        # 下采样阶段的csp
        self.P3_P4_csp = CSPLayer(int(2 * width * ch_in[0]), int(width * ch_in[1]),
                                  repeat=round(3 * depth), shortout=False, dw=dw, activation=activation)
        self.down_conv2 = Conv(int(width * ch_in[1]), int(width * ch_in[1]), k_size=3, stride=2, act_name=activation)

        self.P4_P5_csp = CSPLayer(int(2 * width * ch_in[1]), int(width * ch_in[2]),
                                  repeat=round(3 * depth), shortout=False, dw=dw, activation=activation)

    def forward(self, inputs):
        out_feats = self.backbone.forward(inputs)
        # shape分别为256,80, 80|| 512,40,40 || 1024,20,20
        feat1, feat2, feat3 = [out_feats[f] for f in self.in_feats]

        P5 = self.last_conv(feat3)  # 1024,20,20 => 512,20,20
        P5_upsample = self.upsample(P5)  # 512,20,20 => 512,40,40

        P5_P4_cat = torch.cat([P5_upsample, feat2], dim=1)  # P5的上采样与dark4的输出拼接, =>1024,40,40
        P4 = self.P4_P3_csp(P5_P4_cat)  # 拼接后还有经过csp模块,1024,40,40 => 512,40,40
        P4 = self.P4_conv(P4)  # 512,40,40 => 256,40,40

        P4_upsample = self.upsample(P4)  # 256,80,80
        P4_P3_cat = torch.cat([P4_upsample, feat1], dim=1)  # =>512,80,80

        P3_out = self.P3_csp(P4_P3_cat)  # 加强网络最终输出之一 512,80,80 => 256,80,80

        P3_down = self.down_conv1(P3_out)  # 下采样, 256,80,80 => 256,40,40
        P3_P4_cat = torch.cat([P3_down, P4], dim=1)  # 下采样阶段的拼接, =>512,40,40
        P4_out = self.P3_P4_csp(P3_P4_cat)  # 加强网络最终输出之一 512,40,40 => 512,40,40

        P4_down = self.down_conv2(P4_out)  # 512,40,40 => 512,20,20
        P4_P5_cat = torch.cat([P4_down, P5], dim=1)  # => 1024,20,20

        P5_out = self.P4_P5_csp(P4_P5_cat)  # 加强网络最终输出之一 1024,20,20 => 1024,20,20

        # shape分别为(256,80,80),(512,40,40),(1024,20,20)
        return P3_out, P4_out, P5_out


class Head(nn.Module):
    '''
    解耦头,yolox中,类别预测是单独预测的
    '''

    def __init__(self, num_classes, width=1.0, ch_in=(256, 512, 1024), dw=False, activation='silu'):
        super(Head, self).__init__()
        Conv = DwConvBnAct if dw else ConvBnAct

        # 次序必须与原始模型次序一致
        self.cls_conv, self.box_conv, \
        self.cls_pred, self.box_pred, self.conf_pred, self.stems = [nn.ModuleList() for _ in range(6)]

        for i in range(len(ch_in)):
            self.stems.append(ConvBnAct(int(width * ch_in[i]),
                                        int(width * 256), k_size=1, stride=1, act_name=activation))

            # 两次3x3卷积,提取特征,cls_conv与box_conv结构一样
            self.cls_conv.append(
                nn.Sequential(Conv(int(width * 256), int(width * 256), k_size=3, stride=1, act_name=activation),
                              Conv(int(width * 256), int(width * 256), k_size=3, stride=1, act_name=activation)))
            # 将通道数调整到类别数,使用原始的卷积获得类别预测
            self.cls_pred.append(nn.Conv2d(in_channels=int(width * 256),
                                           out_channels=num_classes,
                                           kernel_size=(1, 1),
                                           stride=(1, 1),
                                           padding=(0, 0)))
            self.box_conv.append(
                nn.Sequential(Conv(int(width * 256), int(width * 256), k_size=3, stride=1, act_name=activation),
                              Conv(int(width * 256), int(width * 256), k_size=3, stride=1, act_name=activation)))
            # 框预测与置信度预测
            self.box_pred.append(nn.Conv2d(in_channels=int(width * 256),
                                           out_channels=4,
                                           kernel_size=(1, 1),
                                           stride=(1, 1),
                                           padding=(0, 0)))
            self.conf_pred.append(nn.Conv2d(in_channels=int(width * 256),
                                            out_channels=1,
                                            kernel_size=(1, 1),
                                            stride=(1, 1),
                                            padding=(0, 0)))

    def forward(self, inputs):
        '''
        :param inputs: list,[[256,80,80],[512,40,40],[1024,20,20]]
        :return:
        '''
        outputs = []
        for i, x in enumerate(inputs):
            x = self.stems[i](x)
            cls_feat = self.cls_conv[i](x)
            cls_pred = self.cls_pred[i](cls_feat)  # num_classes,20,20

            box_feat = self.box_conv[i](x)
            box_pred = self.box_pred[i](box_feat)  # 4,20,20
            conf_pred = self.conf_pred[i](box_feat)  # 1,20,20

            # 框坐标,置信度,类别
            output = torch.cat([box_pred, conf_pred, cls_pred], dim=1)
            outputs.append(output)

        return outputs


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        '''
        :param num_classes:
        :param phi: 当为true时,使用可分离卷积,获得小模型
        '''
        super(YoloBody, self).__init__()
        depth_dic = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
        width_dic = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        depth, width = depth_dic[phi], width_dic[phi]
        depth_wise = True if phi == 'nano' else False

        self.backbone = PANet(depth, width, dw=depth_wise)
        self.head = Head(num_classes, width, dw=depth_wise)

    def forward(self, inputs):
        pa_outs = self.backbone.forward(inputs)
        outputs = self.head.forward(pa_outs)

        return outputs


# 可否以函数的形式定义模型?
# todo 似乎不太行,因为summary时才传入input_size
def yolo_body(num_classes, phi, inputs):
    depth_dic = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
    width_dic = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
    depth, width = depth_dic[phi], width_dic[phi]
    depth_wise = True if phi == 'nano' else False

    pa_outs = PANet(depth, width, dw=depth_wise)(inputs)
    outputs = Head(num_classes, width, dw=depth_wise)(pa_outs)

    return outputs


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloBody(80, 'l').to(device)
    # summary(model, input_size=(3, 640, 640))
    torch.manual_seed(2)
    a = torch.rand([2, 3, 640, 640]).to(device)
    print(a[0])
    print(model(a))

# -*- coding:utf-8 -*-
import os
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
import argparse
from tqdm import tqdm

from utils_func.general import get_classes
from nets.yolox_model import YoloBody, weights_init
from losses.yolox_loss import YoloXLoss
from utils_func.dataloader import YoloDataset, yolo_dataset_collate

# 初始化分布式系统
try:
    world_size = int(os.environ['WORLD_SIZE'])  # 分布式系统上所有节点上所有进程数总和, 一般有多少gpu就有多少进程数
    rank = int(os.environ['RANK'])  # 分布式系统上当前进程号,[0,word_size)
    dist_url = f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
    # dist.init_process_group(backend='nccl',
    #                         init_method=dist_url,
    #                         rank=rank,
    #                         world_size=world_size)
except KeyError:
    world_size = 1
    rank = 0
    # dist.init_process_group(backend='nccl',
    #                         init_method='tcp://127.0.0.1:12584',
    #                         rank=rank,
    #                         world_size=world_size)


def split_datasets(train_ann_path, val_ann_path, num_classes, delta_epoch):
    '''
    划分训练集与测试集
    :param train_ann_path:
    :param val_ann_path: 训练数据路径
    :return:
    '''
    with open(train_ann_path) as f:
        train_lines = f.readlines()
    with open(val_ann_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    train_dataset = YoloDataset(train_lines, cfg['input_shape'],
                                num_classes, delta_epoch, mosaic=cfg['mosaic'], train=True)
    val_dataset = YoloDataset(val_lines, cfg['input_shape'],
                              num_classes, delta_epoch, mosaic=False, train=False)
    train_gen = DataLoader(train_dataset, shuffle=False, batch_size=cfg['batch_sz'],
                           num_workers=4, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
    val_gen = DataLoader(val_dataset, shuffle=False, batch_size=cfg['batch_sz'],
                         num_workers=4, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

    return train_gen, val_gen, num_train, num_val


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main(args):
    ''''''
    '''
    启用分布式进程后,会在每一个GPU上都启动一个main函数
    '''
    # 不同节点上gpu编号相同,设置当前使用的gpu编号
    # args.local_rank接受的是分布式launch自动传入的参数local_rank, 针对当前节点来说, 指每个这个节点上gpu编号
    torch.cuda.set_device(args.local_rank)
    cfg = get_config(args.config)

    # 递归创建多级目录, 若目录已经存在,则不再重复创建, 也不抛异常
    if rank == 0:  # 只在分布式系统第0个进程上创建记录日志文件
        os.makedirs(cfg.output, exist_ok=True)
        init_logging(cfg.output)
    summary_writer = SummaryWriter(log_dir=str(Path(cfg.output) / 'tensorboard')) if rank == 0 else None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open('config.yaml', encoding='utf-8', errors='ignore') as f:
        cfg = yaml.safe_load(f)

    class_names, num_classes = get_classes(cfg['classes_path'])
    model = YoloBody(num_classes, cfg['phi'])

    # 初始化模型权重
    weights_init(model)

    if cfg['model_path']:
        pass

    model_train = model.train().to(device)
    loss_func = YoloXLoss(num_classes)
    model_path = 'weights/yolox_s.pth'
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_train.load_state_dict(torch.load(model_path), strict=False)
        # model_dict = model_train.state_dict()
        # for k,v in model_dict.items():
        #     print(k)
        # print('model_dict=', [k for k,v in model_dict.items()])
        # pretrained_dict = torch.load(model_path, map_location=device)
        # print('pretrained_dict1=',pretrained_dict)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        # print('pretrained_dict2=',pretrained_dict)
        # model_dict.update(pretrained_dict)
        # print('dict2=',model_dict)
        # model_train.load_state_dict(model_dict)
        # print('-==-=============================')
        # print(model_train)

    if torch.cuda.is_available():
        pass

    # train_annotation_path = '2007_train.txt'
    # val_annotation_path = '2007_val.txt'
    # with open(train_annotation_path) as f:
    #     train_lines = f.readlines()
    # with open(val_annotation_path) as f:
    #     val_lines = f.readlines()
    # num_train = len(train_lines)
    # num_val = len(val_lines)

    # 设置每个batch的图片数量
    train_step, val_step = num_train // cfg['batch_sz'], num_val // cfg['batch_sz']

    # 排除数量太少的异常情况,若图片数量连一个batch都没有,不训练了
    if train_step == 0:
        raise ValueError('数据集太小,无法训练')

    for i in range(2):
        if i == 0:  # 冻结训练
            lr, is_freeze, epoch1, epoch2 = 1e-3, False, 0, 3
        else:
            lr, is_freeze, epoch1, epoch2 = 1e-4, True, 3, 6
        optimizer = optim.Adam(model_train.parameters(), lr=lr, weight_decay=5e-4)
        # 设置步长更新方式
        if cfg['cosine']:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        # 是否解冻训练
        for param in model_train.backbone.parameters():
            param.requires_grad = is_freeze

        # 数据增强, 找时间放到其他位置
        train_dataset = YoloDataset(train_lines, cfg['input_shape'],
                                    num_classes, epoch2 - epoch1, mosaic=cfg['mosaic'], train=True)
        val_dataset = YoloDataset(val_lines, cfg['input_shape'],
                                  num_classes, epoch2 - epoch1, mosaic=False, train=False)
        train_gen = DataLoader(train_dataset, shuffle=False, batch_size=cfg['batch_sz'],
                               num_workers=4, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
        val_gen = DataLoader(val_dataset, shuffle=False, batch_size=cfg['batch_sz'],
                             num_workers=4, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

        for epoch in range(epoch1, epoch2):
            train_loss, val_loss = 0, 0
            model_train.train()
            with tqdm(total=train_step, desc=f"Epoch {epoch + 1}/{epoch2}", postfix=dict, mininterval=0.3) as bar:
                # 生成器每次迭代生成batch_size张图片和标签,直到把数据集迭代完.共迭代len(datasets)//batch_size 轮.
                for j, (images, box_data) in enumerate(train_gen):
                    with torch.no_grad():
                        img_tensor = torch.from_numpy(images).to(device)
                        labels = [torch.from_numpy(ann).float().to(device) for ann in box_data]

                    optimizer.zero_grad()  # 清零梯度
                    outputs = model_train(img_tensor)  # 前向传播
                    '''
                    import numpy as np
                    # 用于传入与标注代码一样的输入
                    outputs, labels = [], []
                    for i in range(0, 3):
                        temp = np.load(f'output{i}.npy')
                        temp = torch.from_numpy(temp).cuda()
                        outputs.append(temp)
                    for i in range(0, 4):
                        temp = np.load(f'target{i}.npy')
                        temp = torch.from_numpy(temp).cuda()
                        labels.append(temp)
                    '''
                    loss_value = loss_func(outputs, labels)  # 计算损失
                    loss_value.backward()  # 反向传播
                    optimizer.step()  # 更新梯度

                    train_loss += loss_value.item()  # 合并总损失
                    bar.set_postfix(**{'train_loss': train_loss / (j + 1), 'lr': get_lr(optimizer)})
                    bar.update(1)

            model_train.eval()  # 模型测试,
            # 不把测试显示进度条
            for _, (images, box_data) in enumerate(val_gen):
                with torch.no_grad():
                    img_tensor = torch.from_numpy(images).to(device)
                    labels = [torch.from_numpy(ann).float().to(device) for ann in box_data]
                    outputs = model_train(img_tensor)  # 测试阶段,不更新梯度, 直接写在no_grad()中
                    loss_value = loss_func(outputs, labels)
                val_loss += loss_value.item()

            logger.info(f'Epoch:{str(epoch + 1)}/{str(epoch2)} '
                        f'Total Loss: {train_loss / train_step:.3f} || Val Loss: {val_loss / val_step:.3f}')
            torch.save(model.state_dict(),
                       f'logs/ep%{epoch + 1}-loss{train_loss / train_step:.3f}-val_loss{val_loss:.3f}.pth')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())

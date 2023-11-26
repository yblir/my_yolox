# -*- coding:utf-8 -*-
import os
import yaml
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from utils_func.general import get_classes
from nets.yolox_model import YoloBody, weights_init

# from losses.yolox_loss import YoloXLoss
from losses.yolox_loss_better import YoloXLoss
# from losses.yolox_loss_like_v3_test import YoloXLoss

from utils_func.dataloader import YoloDataset, yolo_dataset_collate
from utils_func.callbacks import LossHistory
from utils_func.general import increment_path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('config.yaml', encoding='utf-8', errors='ignore') as f:
    yaml_cfg = yaml.safe_load(f)


def split_datasets(num_classes, epoch_length):
    '''
    划分训练集与测试集
    :param num_classes: 类别数量
    :param epoch_length: 一个epoch长度
    :return:
    '''
    with open(yaml_cfg['train_ann_path']) as f:
        train_lines = f.readlines()
    with open(yaml_cfg['val_ann_path']) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    train_dataset = YoloDataset(train_lines, yaml_cfg['input_shape'],
                                num_classes, epoch_length, mosaic=yaml_cfg['mosaic'], train=True)
    val_dataset = YoloDataset(val_lines, yaml_cfg['input_shape'],
                              num_classes, epoch_length, mosaic=False, train=False)
    train_gen = DataLoader(train_dataset, shuffle=False, batch_size=yaml_cfg['batch_sz'],
                           num_workers=4, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
    val_gen = DataLoader(val_dataset, shuffle=False, batch_size=yaml_cfg['batch_sz'],
                         num_workers=4, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

    return train_gen, val_gen, num_train, num_val


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    class_names, num_classes = get_classes(yaml_cfg['classes_path'])
    model = YoloBody(num_classes, yaml_cfg['phi'])

    # 初始化模型权重
    weights_init(model)
    if torch.cuda.is_available():
        cudnn.benchmark = True
    model_train = model.train().to(device)
    loss_func = YoloXLoss(num_classes)
    save_dir = increment_path('runs/train/logs')
    loss_history = LossHistory(save_dir, model_train, yaml_cfg['input_shape'])

    if yaml_cfg['resume'] != '':  # 加载预训练模型
        logger.info('Load weights {}.'.format(yaml_cfg['model_path']))
        # model_train.load_state_dict(torch.load(model_path), strict=False)

        model_dict = model_train.state_dict()
        # 此时model_list保存的都是键名,没有值
        model_list = list(model_dict)
        # 加载预训练权重,也是一个有序字典
        pre_dict = torch.load(yaml_cfg['model_path'], map_location=device)
        # 重新更新预训练模型,因为自己搭建的模型部分属性名与原始模型不同,所以不能直接加载,需要把预训练的键名替换成自己的
        # 以下是根据每层参数的shape替换原来的键名.如果构建的模型层次序或shape与原始模型不一致, 莫得法,神仙也搞不定~
        pre_dict = {model_list[i]: v for i, (k, v) in enumerate(pre_dict.items())
                    if np.shape(model_dict[model_list[i]]) == np.shape(v)}
        # 使用更新后的预训练权重,更新本模型权重
        model_dict.update(pre_dict)
        # 加载模型权重字典
        model.load_state_dict(model_dict)

    for i in range(2):
        if i == 0:  # 冻结训练
            lr, is_freeze, epoch1, epoch2 = 1e-3, False, 0, 10
            logger.info('starting freeze train...')
        else:  # 解冻训练
            lr, is_freeze, epoch1, epoch2 = 1e-4, True, 10, 6
            logger.info('starting unfreeze train...')
        optimizer = optim.Adam(model_train.parameters(), lr=lr, weight_decay=5e-4)
        # 设置步长更新方式
        if yaml_cfg['cosine']:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
        # 是否解冻训练
        for param in model_train.backbone.parameters():
            param.requires_grad = is_freeze

        train_gen, val_gen, num_train, num_val = split_datasets(num_classes, epoch_length=epoch2 - epoch1)
        # 设置每个batch的图片数量
        train_step, val_step = num_train // yaml_cfg['batch_sz'], num_val // yaml_cfg['batch_sz']

        # 排除数量太少的异常情况,若图片数量连一个batch都没有,不训练了
        if train_step == 0 or val_step == 0:
            raise ValueError('数据集太小,无法训练')

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
                    # for i, output in enumerate(outputs):
                    #     np.save(f"output{i}", output.detach().cpu().numpy())
                    # for i, label in enumerate(labels):
                    #     np.save(f"label{i}", label.detach().cpu().numpy())
                    outputs, labels = [], []
                    for i in range(0, 3):
                        temp = np.load(f'output{i}.npy')
                        temp = torch.from_numpy(temp).cuda()
                        outputs.append(temp)
                    for i in range(0, 4):
                        temp = np.load(f'label{i}.npy')
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

            # 更新步长
            lr_scheduler.step()

            loss_history.append_loss(epoch + 1, train_loss / train_step, val_loss / val_step)
            logger.info(f'Epoch:{str(epoch + 1)}/{str(epoch2)} '
                        f'Total Loss: {train_loss / train_step:.3f} || Val Loss: {val_loss / val_step:.3f}')
            torch.save(model.state_dict(),
                       f'{save_dir}/ep%{epoch + 1}-loss{train_loss / train_step:.3f}-val_loss{val_loss:.3f}.pth')


if __name__ == '__main__':
    main()

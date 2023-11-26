# -*- coding:utf-8 -*-
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from losses.box_loss import IouLoss
from utils_func.utils_bbox import boxes_iou

np.set_printoptions(threshold=np.inf)


class YoloXLoss(nn.Module):
    def __init__(self, num_classes, strides=(8, 16, 32)):
        super(YoloXLoss, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.box_weight = 5.0
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.iou_loss = IouLoss(reduction='none')
        # self.iou_loss=AlphaIouLoss(reduction='none')
        self.grids = [torch.zeros(1)] * len(strides)  # 占位符

    def get_output_grid(self, output, k, stride):
        '''
        获得
        :param output: [bs,num_classes + 5,80,80],[bs,25,40,40],[bs,25,20,20],每次迭代传入一个
        :param k: 步长或预测输出的序号
        :param stride: 步长,是8,16,32样的整数, 与各自的特征尺寸相乘可得原始图片大小
        :return:
        '''
        feat_h, feat_w = output.shape[-2:]
        # x:横轴重复, y:垂直方向重复   pytorch 1.90以后不支持 indexing='ij' ?
        x, y = torch.meshgrid([torch.arange(feat_h), torch.arange(feat_w)])
        # 构建网格,变换维度并把类型转为和输出一样.两个值的排列次序为 x , y
        grid = torch.stack([y, x], dim=2).reshape(1, feat_h, feat_w, 2).type_as(output)
        self.grids[k] = grid
        grid = grid.reshape(1, -1, 2)  # 将网格拍平成二维,对80x80特征特征层:grid.shape=(1,6400,2)
        output = output.flatten(start_dim=2).permute(0, 2, 1)  # 将输出的特征也拍平 (bs,6400,25)

        # (预测框相对真实框偏移量+预测框本身所在网格坐标值)*该特征相相对原始图片大小倍数.
        # 此时output[...,:2]就是在原始640x640图片大小的预测的xy坐标 和宽高
        # todo 另外,这里不使用sigmoid约束,不会超出单元格本身吗?
        output[..., :2] = (output[..., :2] + grid) * stride
        # 因为没有使用anchor框,所以这里对预测的宽高直接指数处理,得到预测框宽高. 最终该特征成也只会输出一个预测框
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        return output, grid

    @staticmethod
    def in_boxes_info(true_boxes, expanded_strides, x_shifts, y_shifts, total_anchors_num, gt_num, radius=2.5):
        '''
        计算哪些网格点在真实标注框内或其镜像框内(任意一个真实框或镜像框都行), 候选正样本
        计算哪些上述被选中网格点既在真实框又在其镜像框内(在它负责的真实框和镜像框) 最终正样本
        :param true_boxes:
        :param expanded_strides:
        :param x_shifts:
        :param y_shifts:
        :param total_anchors_num:每张图片预测框数量
        :param gt_num:每张图真实框数量
        :param radius:
        :return:
        '''
        # 1,8400=>8400,,每个预测框扩展到原始图片上所需的倍数
        expand_strides = expanded_strides[0]
        # +0.5=每个格子x方向中心,*expand_strides直接将坐标映射为原始图片上坐标.
        # 行方向重复gt_num次,是为了让每个格子的预测框都与真实框比较
        x_center = ((x_shifts[0] + 0.5) * expand_strides).unsqueeze(0).repeat(gt_num, 1)  # [n,8400] n每张图片标注框数量
        y_center = ((y_shifts[0] + 0.5) * expand_strides).unsqueeze(0).repeat(gt_num, 1)

        # 真实框xywh=>x1,y1,x2,y2
        # [gt_num,8400]  左上角xy
        boxes_min = (true_boxes[:, :2] - true_boxes[:, 2:] / 2).unsqueeze(1).repeat(1, total_anchors_num, 1)
        # [gt_num,8400]  右下角xy
        boxes_max = (true_boxes[:, :2] + true_boxes[:, 2:] / 2).unsqueeze(1).repeat(1, total_anchors_num, 1)
        # todo 不用repeat,用广播机制可行不?
        # boxes_min = (true_boxes[:, :2] - true_boxes[:, 2:] / 2)
        # boxes_max = (true_boxes[:, :2] + true_boxes[:, 2:] / 2)
        # 计算每个网格中心点到真实框四边长的距离
        delta_x1 = x_center - boxes_min[:, :, 0]
        delta_y1 = y_center - boxes_min[:, :, 1]
        delta_x2 = boxes_max[:, :, 0] - x_center
        delta_y2 = boxes_max[:, :, 1] - y_center
        delta_box = torch.stack([delta_x1, delta_y1, delta_x2, delta_y2], dim=2)  # [gt_num,8400,4]
        # print('==================')
        # print(delta_box[:6400].detach().cpu().numpy())
        # 如果特征图网格点在真实框内部,那么上述差值的最小也大于0
        is_in_boxes = delta_box.min(dim=-1).values > 0.0
        # print(f'is_in_boxes sum={sum(sum(is_in_boxes.detach().cpu().numpy()))}')
        # (8400,),若特征图上某个网格点落在该图片任何一个真实框内,也算它过关
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # print(f'is_in_boxes_all sum={sum(is_in_boxes_all.detach().cpu().numpy())}')
        # 计算真实框的镜像框,即相对原来的框中心点不变,可能比原框大,也可能更小. 这样是为了挑选更多的正样本
        mirror_min_x = true_boxes[:, 0:1].repeat(1, total_anchors_num) - radius * expand_strides
        mirror_min_y = true_boxes[:, 1:2].repeat(1, total_anchors_num) - radius * expand_strides
        mirror_max_x = true_boxes[:, 0:1].repeat(1, total_anchors_num) + radius * expand_strides
        mirror_max_y = true_boxes[:, 1:2].repeat(1, total_anchors_num) + radius * expand_strides
        # mirror_max = true_boxes[:, :2].unsqueeze(1).repeat(1, total_anchors_num, 1) + radius * expand_strides

        # 计算每个网格中心与镜像框的四边的距离
        delta2_x1 = x_center - mirror_min_x
        delta2_y1 = y_center - mirror_min_y
        delta2_x2 = mirror_max_x - x_center
        delta2_y2 = mirror_max_y - y_center
        delta2_box = torch.stack([delta2_x1, delta2_y1, delta2_x2, delta2_y2], dim=2)  # [gt_num,8400,4]

        is_in_mirror = delta2_box.min(dim=-1).values > 0.0
        is_in_mirror_all = is_in_mirror.sum(dim=0) > 0  # (8400,) [True,False,False,...]
        # print(f'is_in_boxes_all sum={sum(is_in_mirror_all.detach().cpu().numpy())}')
        # (8400,) 逻辑或,看该网格点是否在真实框或其镜像框内
        is_in_union = is_in_boxes_all | is_in_mirror_all
        # (gt_num,二者交集数量), 看被选中的网格点是否同时在其负责的真实框和其镜像框内, 若是,那么这个网格点就很重要,更应该作为正样本
        is_in_inter = is_in_boxes[:, is_in_union] & is_in_mirror[:, is_in_union]
        # print(f"is_in_inter={is_in_inter.shape}")
        return is_in_union, is_in_inter

    @staticmethod
    def dynamic_k_matching(cost, iou, true_cls, gt_num_per_img, is_in_union):
        '''
        simOTA,动态匹配正负样本,是yolox本身一大亮点
        :param cost: (gt_num,is_in_union)
        :param iou: shape同上
        :param true_cls: (n,)
        :param gt_num_per_img:
        :param is_in_union:
        :return:
        '''
        matching_matrix = torch.zeros_like(cost)
        # 原代码只提取10个正样本框,这里的数量不一定是10个
        candidate_k = min(10, iou.shape[1])
        iou_topk, _ = torch.topk(iou, k=candidate_k, dim=1)  # (gt_num,10)

        # 通过topk动态选择框, 统计每个目标分配的候选框数量.
        # 某个真实框与所有候选正样本预测框的topk个iou值求和,再取整, 获得值就是该真实框应该匹配的正样本数量.(n1,n2,n3)
        dynamic_ks = torch.clamp(iou_topk.sum(dim=1).int(), min=1)

        for i in range(gt_num_per_img):  # 遍历这张图片上每个真实标签
            # largest为False,表示从小到大取值,这里选取k个代价值最小的值. 返回索引位置
            _, pos_index = torch.topk(cost[i], k=dynamic_ks[i].item(), largest=False)
            # 待选择正样本位置标记出来
            matching_matrix[i][pos_index] = 1.0

        # 过滤掉多个真实框共用的候选框,对列求和,若大于1,则证明至少有两个真实框对应一个候选框
        matchinig_num = matching_matrix.sum(dim=0)  # (并集数量,)
        matchinig_bool = matchinig_num > 1  # (n,)  n:候选正样本数量(并集数量) is_in_union
        if matchinig_bool.sum() > 0:  # 存在同列有多个1
            # 找出和大于1的列所在的行索引,即哪个真实框的iou值最小
            _, gt_index = torch.min(cost[:, matchinig_bool], dim=0)
            # 将标记出来的 和大于1的列全部置位0
            matching_matrix[:, matchinig_bool] = 0.0
            # 再把那个真实iou最小的位置设为1
            matching_matrix[gt_index, matchinig_bool] = 1.0

        # 找出存在候选框的列,这是对候选框的二次挑选
        fg_mask_inboxes = matching_matrix.sum(dim=0) > 0.0
        # 一个标量值,有几列存在候选框,总的候选框数量, 最终正样本数量
        fg_num_per_img = fg_mask_inboxes.sum().item()

        # 二次挑选的候选框,赋值给初选. 即从初选中再次挑选用于预测的正样本
        is_in_union[is_in_union.clone()] = fg_mask_inboxes
        # 筛选出有候选框的列，并找出筛选列中最大值索引,(n,),n:二次筛选后候选框 列数量,
        # 每列候选框与哪个真实框匹配的索引
        match_gt_index = matching_matrix[:, fg_mask_inboxes].argmax(0)
        # 按照候选框对应真实框的次序,对类别id重新排序,使其与候选框预测的类别匹配, 即这个候选框应该预测的类别是什么
        match_gt_cls = true_cls[match_gt_index]

        # 筛选已选择的候选框的iou
        match_iou = (matching_matrix * iou).sum(dim=0)[fg_mask_inboxes]

        return fg_num_per_img, match_gt_cls, match_iou, match_gt_index

    @torch.no_grad()
    def clc_iou_cost(self, is_in_union, is_in_inter, gt_num,
                     true_boxes, true_cls, pred_boxes_per_img, pred_cls_per_img, pred_conf_per_img):
        '''
        # 计算每张图片gt框与并集候选框iou和cost
        :param is_in_union:
        :param is_in_inter:
        :param gt_num:
        :param true_boxes:
        :param true_cls:
        :param pred_boxes_per_img:
        :param pred_cls_per_img:
        :param pred_conf_per_img:
        :return:
        '''
        # 找出候选正样本的预测框
        pred_boxes_ = pred_boxes_per_img[is_in_union]  # (并集数量,4)
        pred_cls_ = pred_cls_per_img[is_in_union]  # (并集数量,20)
        pred_conf_ = pred_conf_per_img[is_in_union]  # (并集数量,1)
        # 中心点在真实框或其镜像框内的格子数量.也可以成为候选正样本数量
        candidate_post_nums = pred_boxes_.shape[0]

        iou = boxes_iou(true_boxes, pred_boxes_, xyxy=False)
        # 边界框损失,用于计算代价损失值
        iou_loss = -torch.log(iou + 1e-8)

        # [gt_num,is_in_union中True的数量,num_classes]
        pred_cls_ = pred_cls_.float().unsqueeze(0).repeat(gt_num, 1, 1).sigmoid()
        pred_conf_ = pred_conf_.float().unsqueeze(0).repeat(gt_num, 1, 1).sigmoid()
        # 计算类别得分=预测类别值*置信度
        cls_score = pred_cls_ * pred_conf_

        # one_hot编码真实类别值,变成与预测类别shape一样. one_hot必须是torch.int64才能编码,否则报错
        true_cls_one_hot = F.one_hot(true_cls.to(torch.int64), self.num_classes).float()
        true_cls_one_hot = true_cls_one_hot.unsqueeze(dim=1).repeat(1, candidate_post_nums, 1)

        cls_loss = F.binary_cross_entropy(cls_score.sqrt_(), true_cls_one_hot, reduction='none').sum(dim=-1)
        # 每个gt框与所有初选候选框的cost值, 如果不在交集候选框中,cost会很大
        # (gt_num,并集数量),代价损失.公式来自yolox论文. 不同时在真实框与镜像框内的系数设为100000,宣判死刑
        cost = cls_loss + 3.0 * iou_loss + 100000.0 * (~is_in_inter).float()

        return iou, cost

    def clc_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        '''

        :param x_shifts:shape=(1,8400)
        :param y_shifts:同上
        :param expanded_strides: 同上
        :param labels: list,每个元素shape=n,5. n:每张图片人脸框数量. 5:xywh+人脸id
        :param outputs: bs,8400,25
        :return:
        '''
        # shape=8,8400,4  8,8400,1  8,8400,20
        total_pred_boxes, total_pred_conf, total_pred_cls = outputs[..., :4], outputs[..., 4:5], outputs[..., 5:]
        # 8400  原始代码定义为anchor框数量,YOLOx中特征图上每个网格定义为一个anchor
        total_anchors_num = outputs.shape[1]
        # 每个anchor应该预测的真实值
        targets_boxes, targets_conf, targets_cls, fg_masks = [], [], [], []

        batch_fg_num = 0.0  # 总的候选框数量(初选)
        for i in range(outputs.shape[0]):  # 遍历每张图片
            gt_num = len(labels[i])  # 获得每张图片标注框数量
            if gt_num == 0:  # 如果图片没有标注框
                target_cls = outputs.new_zeros((0, self.num_classes))
                target_box = outputs.new_zeros((0, 4))
                target_conf = outputs.new_zeros((total_anchors_num, 1))
                is_in_union = outputs.new_zeros(total_anchors_num).bool()
            else:
                true_boxes = labels[i][..., :4]  # gt_num,4 每张图片真实框坐标,n:每张图片框数量
                true_cls = labels[i][..., 4]  # gt_num, 每个框对应的人脸id
                pred_boxes = total_pred_boxes[i]  # 8400,4 每张图片上的预测框坐标
                pred_conf = total_pred_conf[i]  # 8400,1 每张图片上的预测框的置信度
                pred_cls = total_pred_cls[i]  # 8400,20 每张图片上的预测框的类别id值
                # is_80=pred_boxes[:6400,:].reshape(80,80,-1).detach().cpu().numpy()
                # is_40=pred_boxes[6400:8000].reshape(40,40).detach().cpu().numpy()
                # is_20=pred_boxes[8000:8400].reshape(20,20).detach().cpu().numpy()
                # print('11111111111')
                # print(is_80)
                # print('----------------')
                # print(true_boxes.detach().cpu().numpy())
                # print('================')
                # print(true_cls.detach().cpu().numpy())
                # print('22222222222')
                # print(is_40)
                # print('3333333333')
                # print(is_20)
                # 找出候补正样本并集和交集
                # expanded_strides=expanded_strides[:,8000:]
                # x_shifts=x_shifts[:,8000:]
                # y_shifts=y_shifts[:,8000:]
                # total_anchors_num=400
                is_in_union, is_in_inter = self.in_boxes_info(true_boxes, expanded_strides,
                                                              x_shifts, y_shifts,
                                                              total_anchors_num, gt_num, radius=2.5)
                # is_80=is_in_union.reshape(20,20).detach().cpu().numpy()
                # print('11111111111')
                # print(is_80)
                # print('22222222')
                # print(sum(is_in_union.detach().cpu().numpy()))
                # print(sum(sum(is_in_inter.detach().cpu().numpy())))
                # 计算每张图片gt框与并集候选框iou和cost
                gt_candidate_iou, cost = self.clc_iou_cost(is_in_union, is_in_inter,
                                                           gt_num,
                                                           true_boxes, true_cls,
                                                           pred_boxes, pred_cls, pred_conf)
                # print(gt_candidate_iou.detach().cpu().numpy())
                # print(cost.detach().cpu().numpy())
                # 动态分配正负样本 fg_num_per_img:每张图片最终正样本数量
                fg_num_per_img, match_gt_cls, \
                match_iou, match_gt_index = self.dynamic_k_matching(cost, gt_candidate_iou,
                                                                    true_cls, gt_num, is_in_union)
                # print(match_gt_cls.detach().cpu().numpy())
                # print(match_iou.detach().cpu().numpy())
                # print(match_gt_index.detach().cpu().numpy())
                torch.cuda.empty_cache()  # 这是干嘛的?
                batch_fg_num += fg_num_per_img
                # fg_num_per_img个最终正样本需要学习的类别one_hot向量需乘以其与所学gt的iou。为什么要这么做？似乎很多人都不懂为何这样操作
                # 难道是软标签?, 这个网格应该预测类别one_hot*与其gt框iou,相交多大,类别概率就有多大
                target_cls = F.one_hot(match_gt_cls.to(torch.int64), self.num_classes) * match_iou.unsqueeze(dim=-1)
                # (8400,1),每个格子是否存在物体的置信度,bool值
                target_conf = is_in_union.unsqueeze(dim=-1)
                # 把每个最终正样本需要学习的真实框的xywh抽出来, (n,4),n:正样本数量
                target_box = true_boxes[match_gt_index]
                # print(target_box)
                # is_80=is_in_union[:6400].reshape(80,80).detach().cpu().numpy()
                # is_40=is_in_union[6400:8000].reshape(40,40).detach().cpu().numpy()
                # is_20=is_in_union.reshape(20,20).detach().cpu().numpy()
                # print('11111111111')
                # print(is_80)
                # print(sum(sum(is_80)))
                # print(true_boxes.detach().cpu().numpy())
                # print(true_cls.detach().cpu().numpy())
                # print('22222222222')
                # print(is_40)
                # print('3333333333')
                # print(is_20)

            targets_boxes.append(target_box)
            targets_conf.append(target_conf.type_as(target_cls))
            targets_cls.append(target_cls)
            # 并集所对应的候选框
            fg_masks.append(is_in_union)

        targets_boxes = torch.cat(targets_boxes, dim=0)
        targets_conf = torch.cat(targets_conf, dim=0)
        targets_cls = torch.cat(targets_cls, dim=0)
        fg_masks = torch.cat(fg_masks, dim=0)

        # 计算损失
        batch_fg_num = max(batch_fg_num, 1)

        loss_box = self.iou_loss(total_pred_boxes.reshape(-1, 4)[fg_masks], targets_boxes).sum()
        loss_conf = self.bce_loss(total_pred_conf.reshape(-1, 1), targets_conf).sum()
        loss_cls = self.bce_loss(total_pred_cls.reshape(-1, self.num_classes)[fg_masks], targets_cls).sum()

        loss = self.box_weight * loss_box + loss_conf + loss_cls
        print(loss_box,loss_conf,loss_cls,loss)
        return loss / batch_fg_num

    '''
    改来改去,最终发现,还是把一张图片的输出拍平了合并在一起最简便,运算量最少
    '''
    def forward(self, predict, labels):
        '''
        :param predict: [[bs,num_classes + 5,80,80],[bs,85,40,40],[bs,85,20,20]]
        :param labels: 同上
        :return:
        '''
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        import time
        t1=time.time()
        for i, output in enumerate(predict):
            output, grid = self.get_output_grid(output, i, self.strides[i])
            x_shifts.append(grid[..., 0])  # 水平方向 shape=[[1,6400],[1,1600],[1,400]]
            y_shifts.append(grid[:, :, 1])  # 垂直方向
            # value=[[8...,],[16..,16..],[32...,32...]],shape同上.构建这个变量是干嘛的?
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * self.strides[i])
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)  # shape=bs,8400,25
        x_shifts = torch.cat(x_shifts, dim=1)  # 1,8400
        y_shifts = torch.cat(y_shifts, dim=1)  # 1,8400
        expanded_strides = torch.cat(expanded_strides, dim=1)  # 1,8400

        loss = self.clc_losses(x_shifts, y_shifts, expanded_strides, labels, outputs)
        t2=time.time()
        print(f'耗时={t2-t1}')
        sys.exit()
        return loss

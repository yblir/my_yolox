import numpy as np
import torch
from torchvision.ops import boxes


def xyxy2xywh(coord):
    '''
    把xyxy格式坐标转化为xywh格式.
    coord: x1,y1,x2,y2
    normalized: 是否归一化
    '''
    box_xy = (coord[..., 0:2] + coord[..., 2:4]) / 2
    box_wh = coord[..., 2:4] - coord[..., 0:2]

    coord[..., :2] = box_xy
    coord[..., 2:4] = box_wh

    return coord


def xywh2xyxy(coord):
    '''
    把xywh格式坐标转化为xyxy格式.
    coord: x1,y1,x2,y2
    '''
    box_min = coord[:, :2] - coord[:, 2:4] / 2
    box_max = coord[:, :2] + coord[:, 2:4] / 2

    coord[..., :2] = box_min
    coord[..., 2:4] = box_max

    return coord


def correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        # 这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        # new_shape指的是宽高缩放情况
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2],
                            box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def decode_outputs(outputs, input_shape):
    '''
    解码输出框,与yolox_loss/get_output_gridz中的解码略有不同,
    outputs:[[bs,85,80,80],...]
    input_shape:[640,640],h,w
    '''
    grids, strides = [], []
    # [(80,80),(40,40),(20,20)]
    hw = [x.shape[-2:] for x in outputs]
    # batch_size,8400,5+num_classes,8400分别为80,40,20特征图大小的网格数量. 6400,1600,400
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    # 在训练时没做sigmoid,在预测时却做了这样的处理
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    for h, w in hw:
        # 新版本pytorch 不支持indexing='ij' ?
        # grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='ij')
        grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
        # 1,6400,2
        grid = torch.stack((grid_x, grid_y), dim=2).reshape(1, -1, 2)
        shape = grid.shape[:2]  # 1,6400

        grids.append(grid)
        # (1,6400,1),[[[8]],[[8]],...]
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    # 1,8400,2
    grids = torch.cat(grids, dim=1).type_as(outputs)
    # 1,8400,1
    strides = torch.cat(strides, dim=1).type_as(outputs)
    # 根据网格点进行解码
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    # 归一化,x,w同方向,都是水平方向
    outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
    outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]

    return outputs


def non_max_suppression(predict, num_classes, input_shape,
                        image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    '''
    非极大抑制,过滤重合度大的框
    predict:[batch_size, num_anchors, 85], num_anchors:8400
    num_classes:80
    input_shape:(640,640),输入模型的的尺寸
    image_shape:(h,w) 原始图片尺寸
    letterbox_image:bool,是否对原图进行不失真的缩放
    conf_thres:是否包含物体的置信度,值越大,最终找出的物体越少
    nms_thres:预测框重合程度. 如当值大于0.4时,才把多出的框清理掉.该值越大,保留的预测框越多.
    '''
    # xywh->xyxy
    box_min = predict[..., :2] - predict[..., 2:4] / 2
    box_max = predict[..., :2] + predict[..., 2:4] / 2
    predict[..., :4] = torch.cat([box_min, box_max], dim=-1)

    output = [None for _ in range(predict.shape[0])]
    # 对输入图片进行循环，一般只会进行一次,因为通常都是都仅预测一张图片
    for i, pred in enumerate(predict):
        # shape都是(num_anchors,1),返回 值,和对应索引位置, 位置即预测的类别编号
        class_conf, class_pred = torch.max(pred[:, 5:5 + num_classes], dim=1, keepdim=True)
        # (8400,),bool,利用置信度进行第一轮筛选,得分=置信度预测值*对应类别预测概率值, 用最终得分与是否有物体的置信度阈值比较
        score_mask = pred[:, 4] * class_conf[:, 0] >= conf_thres

        # [num_anchors,7]：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        detections = torch.cat([pred[:, :5], class_conf, class_pred.float()], dim=1)
        # (n,7),从8400个预测框中挑选出得分大于置信度阈值的框
        detections = detections[score_mask]
        # batched_nms根据每个类别进行过滤，只对同一种类别进行计算IOU和阈值过滤,似乎比nms更符合实际情况
        # nms不区分类别对所有bbox进行过滤。如果有不同类别的bbox重叠的话会导致被过滤掉并不会分开计算。
        nms_index = boxes.batched_nms(boxes=detections[:, :4],
                                      # scores与前面score_mask对应数值一样,前次是为了挑选符合置信度阈值的框,本次是
                                      # 为了过滤重合度大的多余框
                                      scores=detections[:, 4] * detections[:, 5],
                                      idxs=detections[:, 6],  # class_pred
                                      iou_threshold=nms_thres)
        # 预测框的第二轮挑选,过滤重合框
        output[i] = detections[nms_index]

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            # 获得归一化后的xywh
            box_xy = (output[i][:, 0:2] + output[i][:, 2:4]) / 2
            box_wh = output[i][:, 2:4] - output[i][:, 0:2]
            # 将缩放后的图片在放缩回原图
            output[i][:, :4] = correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

    return output


def boxes_iou(boxes_a, boxes_b, xyxy=True):
    '''
    计算iou交并比
    :param boxes_a:n1,4,真实框
    :param boxes_b:n2,4,预测框
    :param xyxy:
    :return:shape=(n1,n2) 表示每个真实框与所有预测框的交并比
    '''
    if boxes_a.shape[1] != 4 or boxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        inter_min = torch.max(boxes_a[:, None, :2], boxes_b[:, :2])
        inter_max = torch.min(boxes_a[:, None, 2:], boxes_b[:, 2:])
        area_a = torch.prod(boxes_a[:, 2:] - boxes_a[:, :2], 1)
        area_b = torch.prod(boxes_b[:, 2:] - boxes_b[:, :2], 1)
    else:  # xywh
        # None: 在指定位置添加一个维度1,相当于unsqueeze()
        # 寻找相交部分的坐标
        inter_min = torch.max((boxes_a[:, None, :2] - boxes_a[:, None, 2:] / 2),
                              (boxes_b[:, :2] - boxes_b[:, 2:] / 2))
        inter_max = torch.min((boxes_a[:, None, :2] + boxes_a[:, None, 2:] / 2),
                              (boxes_b[:, :2] + boxes_b[:, 2:] / 2))
        # 计算交集面积
        area_a = torch.prod(boxes_a[:, 2:], dim=1)
        area_b = torch.prod(boxes_b[:, 2:], dim=1)
    # todo 这个en是干嘛的? 当不相交时用来约束为0吗
    en = (inter_min < inter_max).type_as(inter_min).prod(dim=2)
    area_inter = torch.prod(inter_max - inter_min, dim=2) * en
    iou = area_inter / (area_a[:, None] + area_b - area_inter)

    return iou

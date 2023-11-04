import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, ):
        super(MSEWithLogitsLoss, self).__init__()
    
    def forward(self, logits, targets):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)

        pos_id = (targets==1.0).float()
        neg_id = (targets==0.0).float()
        pos_loss = pos_id * (inputs - targets) ** 2
        neg_loss = neg_id * (inputs) ** 2
        loss = 5.0 * pos_loss + 1.0 * neg_loss

        return loss


def compute_loss(pred_conf, pred_cls, pred_txtytwth, targets):
    batch_size = pred_conf.size(0)

    # 损失函数
    conf_loss_function = MSEWithLogitsLoss()
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.MSELoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    # 预测
    pred_conf = pred_conf[:, :, 0]            # [B, HW, ]
    pred_cls = pred_cls.permute(0, 2, 1)      # [B, num_class, HW]
    pred_txty = pred_txtytwth[:, :, :2]       # [B, HW, 2]
    pred_twth = pred_txtytwth[:, :, 2:]       # [B, HW, 2]

    # 标签
    gt_obj = targets[..., 0]                  # [B, HW, ]
    gt_cls = targets[..., 1].long()                  # [B, HW, ]
    gt_txty = targets[..., 2:4]               # [B, HW, 2]
    gt_twth = targets[..., 4:6]               # [B, HW, 2]
    gt_box_scale_weight = targets[..., 6]     # [B, HW, ]

    # 置信度损失
    conf_loss = conf_loss_function(pred_conf, gt_obj)
    conf_loss = conf_loss.sum() / batch_size

    # 类别损失
    cls_loss = cls_loss_function(pred_cls, gt_cls) * gt_obj
    cls_loss = cls_loss.sum() / batch_size

    # 边界框txty损失
    txty_loss = txty_loss_function(F.sigmoid(pred_txty), gt_txty).sum(-1) * gt_obj * gt_box_scale_weight
    txty_loss = txty_loss.sum() / batch_size

    # 边界框twth损失
    twth_loss = twth_loss_function(pred_twth, gt_twth).sum(-1) * gt_obj * gt_box_scale_weight
    twth_loss = twth_loss.sum() / batch_size

    bbox_loss = txty_loss + twth_loss

    # 总的损失
    total_loss = conf_loss + cls_loss + bbox_loss

    return conf_loss, cls_loss, bbox_loss, total_loss


if __name__ == '__main__':
    pass
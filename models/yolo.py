import torch
import torch.nn as nn
import numpy as np

from .basic import Conv, SPP
from .backbone import build_resnet

from .loss import compute_loss

class myYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5):
        super(myYOLO, self).__init__()
        self.device = device                           # cuda或者是cpu
        self.num_classes = num_classes                 # 目标类别的数量，如20或者80
        self.trainable = trainable                     # 训练时，此参数设为True，否则为False
        self.conf_thresh = conf_thresh                 # 对最终的检测框进行筛选时所用到的阈值
        self.nms_thresh = nms_thresh                   # NMS操作中需要用到的阈值
        self.stride = 32                               # 网络最大的降采样倍数
        self.grid_cell = self.create_grid(input_size)  # 用于得到最终的bbox的参数
        self.input_size = input_size                   # 训练时，输入图像的大小，如416


        # >>>>>>>>>>>>>>>>>>>>>>>>> backbone网络 <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # To do：构建我们的backbone网络
        # self.backbone
        self.backbone, feat_dim = build_resnet("resnet18", pretrained=trainable)


        # >>>>>>>>>>>>>>>>>>>>>>>>> neck网络 <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # To do：构建我们的neck网络
        # self.neck
        self.neck = nn.Sequential(
            SPP(),
            Conv(feat_dim*4, feat_dim, k=1),
        )


        # >>>>>>>>>>>>>>>>>>>>>>>>> detection head网络 <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # To do：构建我们的head网络
        # self.convsets
        self.convsets = nn.Sequential(
            Conv(feat_dim, feat_dim//2, k=1),
            Conv(feat_dim//2, feat_dim, k=3, p=1),
            Conv(feat_dim, feat_dim//2, k=1),
            Conv(feat_dim//2, feat_dim, k=3, p=1)
        )


        # >>>>>>>>>>>>>>>>>>>>>>>>> 预测层 <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.pred = nn.Conv2d(feat_dim, 1 + self.num_classes + 4, 1)


        if self.trainable:
            self.init_bias()

    
    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :1], bias_value)
        nn.init.constant_(self.pred.bias[..., 1:1+self.num_classes], bias_value)
    

    def create_grid(self, input_size):
        """
        用于生成G矩阵，其中每个元素都是特征图上的像素坐标

        TODO:
        生成一个tensor：grid_xy，每个位置的元素是网格的坐标，
            这一tensor将在获得边界框参数的时候会用到。
        """
        # 输入图像的宽和高
        w, h = input_size, input_size
        # 特征图的宽和高
        ws, hs = w // self.stride, h // self.stride
        # 生成网格的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        # 将xy两部分的坐标拼起来：[H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        # [H, W, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)

        return grid_xy
    

    def set_grid(self, input_size):
        """
        用于重置G矩阵
        """
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)


    def decode_boxes(self, pred):
        # 
        """
        将网络输出的tx,ty,tw,th四个量转换成bbox的(x1,y1),(x2,y2)
        """
        output = torch.zeros_like(pred)
        # 得到所有bbox的中心点坐标和宽高
        pred[..., :2] = torch.sigmoid(pred[..., :2]) + self.grid_cell
        pred[..., 2:] = torch.exp(pred[..., 2:])

        # 将所有bbox的中心点坐标和宽高换算成x1y1x2y2形式
        output[..., :2] = pred[..., :2] * self.stride - pred[..., 2:] * 0.5
        output[..., 2:] = pred[..., :2] * self.stride + pred[..., 2:] * 0.5

        return output



    def nms(self, dets, scores):
        # 这是一个最基本的基于python语言的nms操作
        # 这一代码来源于Faster RCNN项目
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                    # bbox的宽w和高h
        order = scores.argsort()[::-1]                   # 按照降序对bbox的得分进行排序

        keep = []                                        # 用于保存经过筛的最终bbox结果
        while order.size > 0:
            i = order[0]                                 # 得到最高的那个bbox
            keep.append(i)                               
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        Input:
            bboxes: [H*W, 4], batch = 1
            scores: [H*W, num_classes], batch = 1
        Output:
            bboxes: [N, 4]
            score: [N, ]
            labels: [N,]
        """

        labels = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), labels)]

        # 首先进行阈值筛选，滤除得分低的检测框
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # 对每一类进行NMS操作
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        # 获得最终的检测结果
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels


    @torch.no_grad()
    def inference(self, x):
        # backbone主干网络
        feat = self.backbone(x)

        # neck网络
        feat = self.neck(feat)

        # detection head网络
        feat = self.convsets(feat)

        # 预测层
        pred = self.pred(feat)

        # 对pred的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # 从pred中分理处objectness预测、class预测、bbox的txtytwth预测
        # objectness预测：[B, H*W, 1]
        conf_pred = pred[..., :1]

        # class预测：[B, H*W, num_cls]
        cls_pred = pred[..., 1:1+self.num_classes]

        # txtytwth预测：[B, H*W, 4]
        txtytwth_pred = pred[..., 1+self.num_classes:]

        # 测试时，默认batch是1，
        # 因此，我们不需要用batch这个维度，用[0]将其取走。
        conf_pred = conf_pred[0] # [H*W, 1]
        cls_pred = cls_pred[0] # [H*W, NC]
        txtytwth_pred = txtytwth_pred[0] # [H*W, 4]

        # 每个边界框的得分
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

        # 解算边界框，并归一化边界框：[H*W, 4]
        bboxes = self.decode_boxes(txtytwth_pred) / self.input_size
        bboxes = torch.clamp(bboxes, 0., 1.)

        # 将预测放在cpu上处理，以便进行后处理
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # 后处理
        bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels


    def forward(self, x, targets=None):
        """
        TODO:
        1.前向推理的代码，主要分为两部分：
        2.训练部分：网络得到obj、cls和txtytwth三个分支的预测，然后计算loss；
        3.推理部分：输出经过后处理得到的bbox、cls和每个bbox的预测得分。
        """
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone主干网络
            feat = self.backbone(x)

            # neck网络
            feat = self.neck(feat)

            # detection head网络
            feat = self.convsets(feat)

            # 预测层
            pred = self.pred(feat)

            # 对pred的size做一些view调整，便于后续的处理
            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            pred = pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            # 从pred中分理处objectness预测、class预测、bbox的txtytwth预测
            # objectness预测：[B, H*W, 1]
            conf_pred = pred[..., :1]

            # class预测：[B, H*W, num_cls]
            cls_pred = pred[..., 1:1+self.num_classes]

            # txtytwth预测：[B, H*W, 4]
            txtytwth_pred = pred[..., 1+self.num_classes:]

            # 计算损失
            (
                conf_loss,
                cls_loss,
                bbox_loss,
                total_loss
            ) = compute_loss(pred_conf=conf_pred,
                             pred_cls=cls_pred,
                             pred_txtytwth=txtytwth_pred,
                             targets=targets)
            
            return conf_loss, cls_loss, bbox_loss, total_loss
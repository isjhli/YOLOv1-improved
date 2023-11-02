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

        # >>>>>>>>>>>>>>>>>>>>>>>>> neck网络 <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # To do：构建我们的neck网络
        # self.neck

        # >>>>>>>>>>>>>>>>>>>>>>>>> detection head网络 <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # To do：构建我们的head网络
        # self.head

        # >>>>>>>>>>>>>>>>>>>>>>>>> 预测层 <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)
    

    def create_grid(self, input_size):
        # To do：
        # 生成一个tensor：grid_xy，每个位置的元素是网格的坐标，
        # 这一tensor将在获得边界框参数的时候会用到。
        

    def set_grid(self, input_size):
        # To do：
        # 用于重置grid_xy


    def decode_boxes(self, pred):
        # 将网络输出的tx,ty,tw,th四个量转换成bbox的(x1,y1),(x2,y2)


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


    def postprocess(self, all_local, all_conf):
        """
        bbox_pred: (N, 4), bsize = 1
        prob_pred: (N, num_classes), bsize = 1
        """
        # 后处理代码


    def forward(self, x, target=None):
        # 前向推理的代码，主要分为两部分：
        # 训练部分：网络得到obj、cls和txtytwth三个分支的预测，然后计算loss；
        # 推理部分：输出经过后处理得到的bbox、cls和每个bbox的预测得分。
        
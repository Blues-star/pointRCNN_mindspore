# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from mimetypes import common_types
import mindspore as ms
from mindspore import nn,ops
from mindspore.ops import functional
from mindspore import Tensor
from mindspore.common.initializer import initializer
# from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModule
from tools.layer_utils import PointnetSAModule
from lib.rpn.proposal_target_layer import ProposalTargetLayer
from tools import layer_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg

import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils


class RCNNNet(nn.Cell):
    def __init__(self, num_classes, input_channels=0, use_xyz=True,mode='TRAIN'):
        super().__init__()

        # self.SA_modules = nn.ModuleList()
        self.SA_modules = []
        channel_in = input_channels
        self.training = (mode=='TRAIN')
        if cfg.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            self.xyz_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel] + cfg.RCNN.XYZ_UP_LAYER,
                                                   bn=cfg.RCNN.USE_BN)
            c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)

        for k in range(cfg.RCNN.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + cfg.RCNN.SA_CONFIG.MLPS[k]

            npoint = cfg.RCNN.SA_CONFIG.NPOINTS[k] if cfg.RCNN.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=npoint,
                    radius=cfg.RCNN.SA_CONFIG.RADIUS[k],
                    nsample=cfg.RCNN.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RCNN.USE_BN
                )
            )
            channel_in = mlps[-1]

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(1-cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.SequentialCell(cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            
            # self.cls_loss_func = F.binary_cross_entropy
            self.cls_loss_func = ops.BinaryCrossEntropy() 
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':

            cls_weight = ms.Tensor.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(weight=cls_weight,ignore_index=-1, reduce=None)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(1-cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.SequentialCell(*reg_layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            raise NotImplementedError
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            # init_func = nn.init.xavier_normal_
            init_func = ms.common.initializer.XavierUniform()
        elif weight_init == 'normal':
            # init_func = nn.init.normal_
            init_func = ms.common.initializer.Normal()
        else:
            raise NotImplementedError

        for m in self.cells():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    # init_func(m.weight, mean=0, std=0.001)
                    init_func = ms.common.initializer.Normal(sigma=0.001)
                    m.weight.set_data(m.weight.clone(init_func))
                else:
                    # init_func(m.weight)
                    m.weight.set_data(m.weight.clone(init_func))
                if m.bias is not None:
                    z = ms.common.initializer.Zero()
                    m.bias.set_data(m.bias.clone(z))
                    # nn.init.constant_(m.bias, 0)
        # normal_ = ms.common.initializer.Normal(sigma=0.001)
        # nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        # xyz = pc[..., 0:3].contiguous()
        xyz = pc[..., 0:3]
        # Tensor.transpose()
        features = (
            pc[..., 3:].swapaxes(1, 2)
            if pc.shape[-1] > 3 else None
        )

        return xyz, features

    def construct(self,**input_data):
        """
        :param input_data: input dict
        :return:
        """
        # print(len(args))
        if cfg.RCNN.ROI_SAMPLE_JIT:
            # True
            if self.training:
                # with torch.no_grad():
                target_dict = self.proposal_target_layer(**input_data)

                pts_input = ops.concat((target_dict['sampled_pts'], target_dict['pts_feature']), axis=2)
                target_dict['pts_input'] = pts_input
            else:
                rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
                batch_rois = input_data['roi_boxes3d']
                if cfg.RCNN.USE_INTENSITY:
                    # False
                    # Tensor.expand_dims()
                    pts_extra_input_list = [input_data['rpn_intensity'].expand_dims(axis=2),
                                            input_data['seg_mask'].expand_dims(axis=2)]
                else:
                    pts_extra_input_list = [input_data['seg_mask'].expand_dims(axis=2)]

                if cfg.RCNN.USE_DEPTH:
                    # True
                    pts_depth = input_data['pts_depth'] / 70.0 - 0.5
                    pts_extra_input_list.append(pts_depth.expand_dims(axis=2))
                pts_extra_input = ops.concat(pts_extra_input_list, axis=2)

                pts_feature = ops.concat((pts_extra_input, rpn_features), axis=2)
                pooled_features, pooled_empty_flag = \
                        roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                      sampled_pt_num=cfg.RCNN.NUM_POINTS)

                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.expand_dims(axis=2)
                for k in range(batch_size):
                    pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                        batch_rois[k, :, 6])

                pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        else:
            pts_input = input_data['pts_input']
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input']
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d']
            if self.training:
                target_dict['cls_label'] = input_data['cls_label']
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask']
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct']

        xyz, features = self._break_up_pc(pts_input)

        if cfg.RCNN.USE_RPN_FEATURES:
            # True
            xyz_input = pts_input[..., 0:self.rcnn_input_channel].swapaxes(1, 2).expand_dims(axis=3)
            xyz_feature = self.xyz_up_layer(xyz_input)

            rpn_feature = pts_input[..., self.rcnn_input_channel:].swapaxes(1, 2).expand_dims(axis=3)

            merged_feature = ops.concat((xyz_feature, rpn_feature), axis=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(axis=3)]
        else:
            l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # Tensor.squeeze()
        rcnn_cls = self.cls_layer(l_features[-1]).swapaxes(1, 2).squeeze(axis=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(l_features[-1]).swapaxes(1, 2).squeeze(axis=1)  # (B, C)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        if self.training:
            ret_dict.update(target_dict)
        return ret_dict


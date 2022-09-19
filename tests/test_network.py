
from logging import Logger
import logging
import mindspore as ms 
from mindspore import Model
import sys,os
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib/datasets'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib/net'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools/'))
from lib.config import cfg, cfg_from_file, save_config_to_file
from mindspore import context
from lib.net.point_rcnn import PointRCNN
from tools.datautil import create_dataloader, create_dataloader_debug
from lib.net.ms_loss import net_with_loss

ms.context.set_context(device_target="GPU")
ms.context.set_context(mode=ms.PYNATIVE_MODE,pynative_synchronize=True)
logger = logging.getLogger("net")
# train_loader, test_loader,num_class = create_dataloader_debug(logger=logger)
# total_step = train_loader.get_dataset_size() * 200

train_mode = 'rpn'
cfg_file = "tools/cfgs/default.yaml"


if cfg_file is not None:
    cfg_from_file(cfg_file)
    print(cfg.RPN.USE_INTENSITY)
    tag = os.path.splitext(os.path.basename(cfg_file))[0]

    if train_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rpn', cfg.TAG)
    elif train_mode == 'rcnn':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
    elif train_mode == 'rcnn_offline':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
    else:
        raise NotImplementedError
# test_data_v2 = {
#     "sample_id":ms.numpy.randint(1,2,(1,)),
#     "pts_input":ms.numpy.rand((1, npoint, 3)),
#     "pts_rect":ms.numpy.rand((1, npoint, 3)),
#     "pts_features":ms.numpy.rand((1, npoint, 1)),
#     "rpn_cls_label":ms.numpy.randint(1,2,(1, npoint)),
#     "rpn_reg_label":ms.numpy.rand((1, npoint, 7)),
#     "gt_boxes3d":ms.numpy.rand((1, 1, 7)),
# }
data,_,num_class = create_dataloader_debug(logger)
data_batch = data.create_dict_iterator().__next__()
print(data_batch.keys())

net = PointRCNN(num_classes=num_class,
                    use_xyz=True,
                    mode='TRAIN')
model = net_with_loss(net)
ans = model(**data_batch)
print(f"loss: {ans}")
print("test passed!")
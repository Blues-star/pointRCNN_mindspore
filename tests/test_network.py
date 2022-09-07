
from logging import Logger
import logging
import mindspore as ms 
import sys,os

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib/datasets'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib/net'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools/'))
from mindspore import context
from lib.net.point_rcnn import PointRCNN
from tools.datautil import create_dataloader, create_dataloader_debug


ms.context.set_context(device_target="GPU")
ms.context.set_context(mode=ms.PYNATIVE_MODE,pynative_synchronize=True)
logger = logging.getLogger("net")
# train_loader, test_loader,num_class = create_dataloader_debug(logger=logger)
# total_step = train_loader.get_dataset_size() * 200
net = PointRCNN(num_classes=2,
                    use_xyz=True,
                    mode='TRAIN')
# train_loader :ms.dataset.GeneratorDataset = train_loader
# test_data = train_loader.create_dict_iterator().__next__()
# for k,v in test_data.items():
#     print(k,v.shape,v.dtype)
npoint = 128
test_data_v2 = {
    "sample_id":ms.numpy.randint(1,2,(1,)),
    "pts_input":ms.numpy.rand((1, npoint, 4)),
    "pts_rect":ms.numpy.rand((1, npoint, 3)),
    "pts_features":ms.numpy.rand((1, npoint, 1)),
    "rpn_cls_label":ms.numpy.randint(1,2,(1, npoint)),
    "rpn_reg_label":ms.numpy.rand((1, npoint, 7)),
    "gt_boxes3d":ms.numpy.rand((1, 1, 7)),
}

net(test_data_v2)
print("test passed!")
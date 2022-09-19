import mindspore as ms
import numpy as np
from mindspore import nn,ops
import lib.utils.iou3d.iou3d_utils as iou3d_utils
ms.context.set_context(device_target="GPU")
ms.context.set_context(mode=ms.PYNATIVE_MODE,pynative_synchronize=True)

pred_reg = ms.numpy.randint(1,64,(64,24)).astype(ms.float32)
x_bin_label = z_bin_label = ms.numpy.randint(1,64,(64,))
cross_entropy = nn.SoftmaxCrossEntropyWithLogits(True,'mean')

t1 = pred_reg[:, 0: 12]
# t1.shape = t1.shape
t2 = pred_reg[:, 12: 24]
# t2.shape = t2.shape
assert ops.Shape()(t1)[-1] > 0
loss_x_bin:ms.Tensor = cross_entropy(t1, x_bin_label)
loss_z_bin:ms.Tensor = cross_entropy(t2, z_bin_label)
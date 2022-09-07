import mindspore as ms
from mindspore import nn,ops
import sys
from pathlib import Path


sys.path.insert(0, Path(__file__).parent.parent.absolute().__str__())
# sys.path.insert(0, '../../')
print(sys.path)
ms.context.set_context(mode=ms.PYNATIVE_MODE)
import lib.net.layer_utils as layer_utils
from lib.utils.roipool3d.roipool3d_utils import roipool3d_gpu,pts_in_boxes3d_cpu, roipool_pc_cpu
from lib.utils.iou3d.iou3d_utils import boxes_iou3d_gpu, boxes_iou_bev, nms_gpu, nms_normal_gpu
# from lib.net.layer_utils import ThreeNN


so_name = "roipool3d_cuda.cpython-39-x86_64-linux-gnu.so"

def roipool_pc_cpu(pts, pts_feature, boxes3d, sampled_pt_num):
    """
    :param pts: (N, 3)
    :param pts_feature: (N, C)
    :param boxes3d: (M, 7)
    :param sampled_pt_num: int
    :return:
    """
    pts = pts.astype(ms.float32)
    pts_feature = pts_feature.astype(ms.float32)
    boxes3d = boxes3d.astype(ms.float32)
    assert pts.shape[0] == pts_feature.shape[0] and pts.shape[1] == 3, '%s %s' % (pts.shape, pts_feature.shape)
    pooled_pts = ms.numpy.zeros((boxes3d.shape[0], sampled_pt_num, 3), dtype=ms.numpy.float32)
    pooled_features = ms.numpy.zeros((boxes3d.shape[0], sampled_pt_num, pts_feature.shape[1]), dtype=ms.numpy.float32)
    pooled_empty_flag = ms.numpy.zeros(boxes3d.shape[0], dtype=ms.numpy.int64)
    roipool3d_cpu = layer_utils.get_func_from_so(so_name,"roipool3d_cpu")
    roipool3d_cpu(pts, boxes3d, pts_feature, pooled_pts, pooled_features, pooled_empty_flag)
    return pooled_pts, pooled_features, pooled_empty_flag


"""
:param pts: (N, 3)
:param pts_feature: (N, C)
:param boxes3d: (M, 7)
:param sampled_pt_num: int
:return:
"""
def test():
    # roipool_pc_cpu()
    N = 64
    C = 3
    M = 16
    sampled_pt_num = 512
    pts = ms.numpy.rand((N,3))
    pts_feature = ms.numpy.rand((N,C))
    boxes3d = ms.numpy.rand((M,3))
    roipool_pc_cpu(pts,pts_feature,boxes3d,sampled_pt_num) 


if __name__ == "__main__":
    test()
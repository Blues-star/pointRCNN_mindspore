import os
from random import random
import mindspore as ms
from mindspore import nn,ops
import sys
from pathlib import Path


sys.path.insert(0, Path(__file__).parent.parent.absolute().__str__())
# sys.path.insert(0, '../../')
print(sys.path)
ms.context.set_context(device_target="GPU")
ms.context.set_context(mode=ms.PYNATIVE_MODE)
import lib.net.layer_utils as layer_utils
from lib.utils.roipool3d.roipool3d_utils import roipool3d_gpu,pts_in_boxes3d_cpu, roipool_pc_cpu
from lib.utils.iou3d.iou3d_utils import boxes_iou3d_gpu, boxes_iou_bev, nms_gpu, nms_normal_gpu
# from lib.net.layer_utils import ThreeNN


def test_3nn():
    # test 3nn
   
    unknown = ms.numpy.rand(16, 1024, 3)
    known = ms.numpy.rand(16, 2048, 3)
    ops = layer_utils.ThreeNN()
    dist, idx = ops(unknown, known)
    print(dist.shape,idx.shape)
    # print(idx.sum())

def test_ThreeInterpolate():
    
    B = 16
    c = 3
    m = 1024
    n = 2048

    features = ms.numpy.rand(B,c,m)
    idx = ms.numpy.rand(B, n, 3).astype(ms.int32)

    weight = ms.numpy.rand(B, n, 3)
    ops = layer_utils.ThreeInterpolate()
    dist = ops(features, idx, weight)
    print(dist.shape)
    # print(idx.sum())

def test_roipool3d_gpu():
    """
    :param pts: (B, N, 3)
    :param pts_feature: (B, N, C)
    :param boxes3d: (B, M, 7)
    :param pool_extra_width: float
    :param sampled_pt_num: int
    :return:
        pooled_features: (B, M, 512, 3 + C)
        pooled_empty_flag: (B, M)
    """
    B = 16
    N = 1024
    C = 3
    M = 128
    pooled_extra_width = 0.1
    sampled_pt_num = 512
    pts = ms.numpy.rand(B, N, 3)
    pts_feature = ms.numpy.rand(B, N, C)
    boxes3d = ms.numpy.rand(B, M, 7)
    pooled_features,pooled_empty_flag = roipool3d_gpu(pts, pts_feature, boxes3d, pooled_extra_width, sampled_pt_num)
    print(pooled_features.shape, pooled_empty_flag.shape)


def test_pts_in_boxes3d_cpu():
    """
    :param pts: (N, 3) in rect-camera coords
    :param boxes3d: (M, 7)
    :return: boxes_pts_mask_list: (M), list with [(N), (N), ..]
    """
    N = 256
    M = 128
    pts = ms.numpy.rand(N, 3)
    boxes3d = ms.numpy.rand(M, 7)
    boxes_pts_mask_list = pts_in_boxes3d_cpu(pts,boxes3d)

    print(boxes_pts_mask_list.shape)
    # print(idx.sum())

def test_roipool_pc_cpu():
    """
    :param pts: (N, 3)
    :param pts_feature: (N, C)
    :param boxes3d: (M, 7)
    :param sampled_pt_num: int
    :return:
    """
    N = 1024
    C = 3
    M = 128
    sampled_pt_num = 512
    pts = ms.numpy.rand(N, 3)
    pts_feature = ms.numpy.rand(N, C)
    boxes3d = ms.numpy.rand(M, 7)
    pooled_pts, pooled_features, pooled_empty_flag = roipool_pc_cpu(pts, pts_feature, boxes3d, sampled_pt_num)
    print(pooled_pts.shape,pooled_features.shape, pooled_empty_flag.shape)

def test_boxes_iou_bev():
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """
    M = 128
    N = 128
    boxes_a = ms.numpy.rand(M, 5)
    boxes_b = ms.numpy.rand(N, 5)
    
    ans_iou = boxes_iou_bev(boxes_a, boxes_b)
    print(ans_iou.shape)

def test_boxes_iou3d_gpu():
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    M = 128
    N = 128
    boxes_a = ms.numpy.rand(N, 7)
    boxes_b = ms.numpy.rand(M, 7)
    ans_iou = boxes_iou3d_gpu(boxes_a, boxes_b)
    print(ans_iou.shape)

def test_nms_gpu():
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    N = 128
    boxes = ms.numpy.rand(N, 5)
    scores = ms.numpy.rand(N)
    thresh = 0.5
    ans = nms_gpu(boxes, scores, thresh)
    print(ans.shape)

def test_nms_normal_gpu():
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    N = 128
    boxes = ms.numpy.rand(N, 5)
    scores = ms.numpy.rand(N)
    thresh = 0.5
    ans = nms_normal_gpu(boxes, scores, thresh)
    print(ans.shape)

def test_grouping_operation():
    B = 2
    C = 3
    N = 2048
    npoint = 4096
    nsample = 16
    features = ms.numpy.rand((B,C,N))
    idx = ms.numpy.randint(1,10,(B,npoint,nsample))
    output = layer_utils.grouping_operation(features,idx)
    print(output.shape)


def test_GatherOperation():
    """
    :param features: (B, C, N)
    :param idx: (B, npoint) index tensor of the features togather int32
    :return:
        output: (B, C, npoint)
    """
    B = 2
    C = 1
    N = 128
    npoint = 64
    features = ms.numpy.rand((B, C, N))
    idx = ms.numpy.randint(1,10,(B, npoint))
    op = layer_utils.GatherOperation()
    out = op(features,idx)
    print(out.shape)


def test_GroupingOperation():
    """
    :param ctx:
    :param features: (B, C, N) tensor of features to group
    :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
    :return:
        output: (B, C, npoint, nsample) tensor
    """
    B = 2
    C = 1
    N = 128
    npoint = 4096
    nsample = 16
    features = ms.numpy.rand((B, C, N))
    idx = ms.numpy.randint(1,10,(B, npoint,nsample))
    output= layer_utils.grouping_operation(features,idx)
    print(output.shape)

def test_BallQuery():
    """
    :param radius: float, radius of the balls
    :param nsample: int, maximum number of features in the balls
    :param xyz: (B, N, 3) xyz coordinates of the features
    :param new_xyz: (B, npoint, 3) centers of the ball query
    :return:
        idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
    """
    radius = 0.5
    nsample = 64
    xyz = ms.numpy.rand(16,512,3)
    new_xyz = ms.numpy.rand(16,256,3)
    idx = layer_utils.ball_query(radius,nsample,xyz,new_xyz)
    print(idx.shape)



if __name__ == "__main__":

    print("test_grouping_operation")
    for i in range(10):
        test_grouping_operation()
    input()
    print("test_BallQuery")
    for i in range(10):
        test_BallQuery()
    print("test_GroupingOperation()")
    for i in range(10):
        test_GroupingOperation()
    input()
    print("test_GatherOperation")
    for _ in range(10):
        test_GatherOperation()
    # print(sys.path)
    print("test_3nn")
    for _ in range(10):
        test_3nn()
    print("test_ThreeInterpolate")
    for _ in range(10):
        test_ThreeInterpolate()
    print("test_roipool3d_gpu")
    test_roipool3d_gpu()
    print("test_pts_in_boxes3d_cpu")
    test_pts_in_boxes3d_cpu()

    print("test_roipool_pc_cpu")
    test_roipool_pc_cpu()

    print("test_boxes_iou_bev")
    test_boxes_iou_bev()
    print("test_boxes_iou3d_gpu")
    test_boxes_iou3d_gpu()

    for i in range(7):
        print("test_nms_gpu")
        test_nms_gpu()

    for i in range(7):
        print("test_nms_normal_gpu",i)
        test_nms_normal_gpu()

    
    print("done")
    
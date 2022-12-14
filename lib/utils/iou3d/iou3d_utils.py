# import torch
# import iou3d_cuda
import lib.utils.kitti_utils as kitti_utils
import mindspore as ms
from mindspore import ops
from pathlib import Path
import os,sys
from pathlib import Path
sys.path.insert(0,Path(__file__).absolute().parent.parent.parent.parent.parent)
from tools.layer_utils import get_func_from_so


so_name = "iou3d_cuda.cpython-39-x86_64-linux-gnu.so"

def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """

    # ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()
    ans_iou = ms.numpy.zeros((boxes_a.shape[0], boxes_b.shape[0]))

    # iou3d_cuda.boxes_iou_bev_gpu(boxes_a, boxes_b, ans_iou)
    # so_name = "iou3d_cuda.cpython-39-x86_64-linux-gnu.so"
    op_boxes_iou_bev_gpu = get_func_from_so(so_name,"boxes_iou_bev_gpu",out_shape=(boxes_a.shape[0], boxes_b.shape[0]),out_dtype=ms.float32)
    ans_iou = op_boxes_iou_bev_gpu(boxes_a, boxes_b)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = kitti_utils.boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = kitti_utils.boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    # overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    # overlaps_bev = ms.numpy.zeros((boxes_a.shape[0], boxes_b.shape[0]))
    # iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev, boxes_b_bev, overlaps_bev)
    boxes_overlap_bev_gpu_op = get_func_from_so(so_name,"boxes_overlap_bev_gpu",out_shape=(boxes_a.shape[0], boxes_b.shape[0]),out_dtype=ms.float32)
    # boxes_overlap_bev_gpu_op(boxes_a_bev, boxes_b_bev, overlaps_bev)
    overlaps_bev = boxes_overlap_bev_gpu_op(boxes_a_bev, boxes_b_bev)
    # height overlap
    boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1)
    boxes_a_height_max = boxes_a[:, 1].view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1)
    boxes_b_height_max = boxes_b[:, 1].view(1, -1)

    max_of_min = ops.maximum(boxes_a_height_min, boxes_b_height_min)
    min_of_max = ops.minimum(boxes_a_height_max, boxes_b_height_max)
    mm = ms.Tensor(0, ms.float32)
    overlaps_h = ops.clip_by_value(min_of_max - max_of_min, clip_value_min=mm)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    mmm = ms.Tensor(1e-7, ms.float32)
    iou3d = overlaps_3d / ops.clip_by_value(vol_a + vol_b - overlaps_3d, clip_value_min=mmm)

    return iou3d


def nms_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = ops.Sort(axis=0,descending=True)(scores)[1]
    # order = scores.sort(0, descending=True)[1]

    boxes = boxes[order]
    assert boxes.dtype == ms.float32
    # keep = torch.LongTensor(boxes.size(0))
    keep = ms.numpy.zeros((boxes.shape[0]),ms.int64)
    num_out:ms.Tensor = ms.numpy.zeros((1), ms.int32)
    # num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh)
    nms_gpu_op = get_func_from_so(so_name,"nms_gpu",out_shape=(1,),out_dtype=ms.int32)
    num_out = nms_gpu_op(boxes, keep, thresh)
    num_out = num_out.asnumpy().item()
    return order[keep[:num_out]]


def nms_normal_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 5
    assert boxes.shape[0] == scores.shape[0]
    # areas = (x2 - x1) * (y2 - y1)
    order = ops.Sort(axis=0,descending=True)(scores)[1]

    boxes = boxes[order]

    # keep = torch.LongTensor(boxes.size(0))
    keep = ms.numpy.zeros((boxes.shape[0]), ms.int64)
    # num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh)
    # num_out:ms.Tensor = ms.numpy.zeros((1), ms.int32)
    thresh = ms.Tensor(thresh,ms.float32)
    in_type = (boxes.shape,keep.shape,thresh.shape)
    print(in_type)
    nms_normal_gpu_op = get_func_from_so(so_name,"nms_normal_gpu",out_shape=(1,),out_dtype=ms.int32,in_type=in_type)
    num_out = nms_normal_gpu_op(boxes, keep, thresh)
    num = num_out.asnumpy().item()
    return order[keep[:num]]


if __name__ == '__main__':
    pass

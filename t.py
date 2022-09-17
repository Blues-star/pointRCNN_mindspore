import mindspore as ms
from mindspore import nn,ops
import torch

# def boxes3d_to_bev_torch(boxes3d):
#     """
#     :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
#     :return:
#         boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
#     """
#     # boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))
#     boxes_bev = ms.numpy.zeros((boxes3d.shape[0], 5))

#     cu, cv = boxes3d[:, 0], boxes3d[:, 2]
#     half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
#     boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
#     boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
#     boxes_bev[:, 4] = boxes3d[:, 6]
#     return boxes_bev

N = 64
boxes3d = ms.numpy.rand((N,7))
mask:ms.Tensor = ms.numpy.randint(0,2,N).astype(ms.bool_)
print(mask.shape)
# ans = ops.masked_select(boxes3d,mask.expand_dims(-1))
ans = boxes3d * mask.expand_dims(-1)
print(ans.shape)
msp_ans = ans.reshape((-1,7)).asnumpy()

box_torch = torch.Tensor(boxes3d.asnumpy())
mask = torch.Tensor(mask.asnumpy()).bool()
torch_ans:torch.Tensor = box_torch[mask]
torch_ans = torch_ans.numpy()

assert (msp_ans == torch_ans).all()==True #True
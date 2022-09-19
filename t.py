import mindspore as ms
from mindspore import nn,ops


a = ms.numpy.randint(0,128,(512,))
print(ops.nonzero(a>256).shape)
print(ops.nonzero(a>256).view(-1))
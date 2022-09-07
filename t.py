import mindspore as ms
from mindspore import nn

ms.context.set_context(device_target="GPU")
ms.context.set_context(mode=ms.PYNATIVE_MODE)
class t(nn.Cell):
    def __init__(self):
        super().__init__()
    
    def construct(self,**kwargs):
        ans = {
            "a":ms.numpy.rand((5,12)),
            "n":ms.numpy.randint(2,13,(512,3))
        }
        return ans


tt = t()
ans = tt()
print(type(ans))
import torch
from mindspore import save_checkpoint
from mindspore import Tensor


params_dict = torch.load('PointRCNN.pth')

ms_param_dict_list = []

for name, param in params_dict['model_state'].items():
    ms_param_dict = {}
    if 'num_batches_tracked' in name:
        continue
    if 'layer0.conv' in name:
        name = name.replace('layer0', 'celllist.0')
    if 'layer1.conv' in name:
        name = name.replace('layer1', 'celllist.1')
    if 'layer2.conv' in name:
        name = name.replace('layer2', 'celllist.2')
    if 'layer0.bn' in name:
        name = name.replace('layer0.bn', 'celllist.0')
    if 'layer1.bn' in name:
        name = name.replace('layer1.bn', 'celllist.1')
    if 'layer2.bn' in name:
        name = name.replace('layer2.bn', 'celllist.2')
    if 'running_mean' in name or 'running_var' in name:
        name = name.replace('running_mean', 'moving_mean')
        name = name.replace('running_var', 'moving_variance')
    if 'bn.weight' in name or 'bn.bias' in name:
        name = name.replace('bn.weight', 'bn.gamma')
        name = name.replace('bn.bias', 'bn.beta')
    if 'rpn.rpn' in name and 'bn' in name:
        name = name.replace('bn.bn.', 'bn.')
    if 'rcnn_net.cls_layer' in name and 'weight' in name:
        param = param[:, :, :, None]
    if 'rpn.rpn_cls_layer' in name and 'weight' in name:
        param = param[:, :, :, None]
    if 'rcnn_net.reg_layer' in name and 'weight' in name:
        param = param[:, :, :, None]
    if 'rpn.rpn_reg_layer' in name and 'weight' in name:
        param = param[:, :, :, None]
    name = '_backbone.' + name
    ms_param_dict['name'] = name
    # print(name, param.shape)
    ms_param_dict['data'] = Tensor(param.cpu().numpy())
    ms_param_dict_list.append(ms_param_dict)

save_checkpoint(ms_param_dict_list, 'PointRCNN.ckpt')
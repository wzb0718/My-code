from .swin_transformer import SwinTransformer
from .resnet14 import ResNet,BasicBlock
from torchvision.ops.misc import FrozenBatchNorm2d
import torch
import os
from .feature_pyramid_network import TwoBackboneCatFpn
from torch import nn
from torch.nn import functional as F


def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps

def resnet_14_cat_swin_transformer_fpn(res_pretrain_path="",
                                       swin_pretrain_path='',
                                       norm_layer=FrozenBatchNorm2d,
                                       res_returned_layers=None,
                                       swin_returned_layers=None,
                                       aggregation='Mean',
                                       input_size=224,
                                       num_classes=3):
    resnet_backbone = ResNet(BasicBlock, [1, 1, 3, 1],
                             include_top=False,
                             norm_layer=norm_layer)
    swin_backbone=SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24))
    # 载入预训练权重
    if res_pretrain_path != "":
        assert os.path.exists(res_pretrain_path), "{} is not exist.".format(res_pretrain_path)
        # 载入预训练权重
        print(resnet_backbone.load_state_dict(torch.load(res_pretrain_path), strict=False))

    if swin_pretrain_path != "":
        assert os.path.exists(swin_pretrain_path), "{} is not exist.".format(swin_pretrain_path)
        # 载入预训练权重
        weights_dict = torch.load(swin_pretrain_path)
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        # ret = model.load_state_dict(weights_dict, strict=False)
        #
        # pretrained_weights = torch.load(swin_pretrain_path)
        # pretrained_weights['pos_drop'] = None
        print(swin_backbone.load_state_dict(weights_dict, strict=False))

    if res_returned_layers is None:
        res_returned_layers = [3,4]
    # 返回的特征层个数肯定大于0小于5
    assert min(res_returned_layers) > 0 and max(res_returned_layers) < 5
    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return_layers_res = {f'layer{k}': str(v) for v, k in enumerate(res_returned_layers)}

    if swin_returned_layers is None:
        swin_returned_layers = [3,4]
    # 返回的特征层个数肯定大于0小于5
    assert min(swin_returned_layers) > 0 and max(swin_returned_layers) < 5
    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return_layers_swin = {f'layers{k}': str(v) for v, k in enumerate(swin_returned_layers)}


    in_channels_list = [160 * 2**(i-1) for i in res_returned_layers]
    # 通过fpn后得到的每个特征层的channel
    out_channels = 256
    return TwoBackboneCatFpn(backbone1=resnet_backbone, backbone2=swin_backbone,return_layers1=return_layers_res,
    return_layers2=return_layers_swin, in_channels_list=in_channels_list, out_channels=out_channels,aggregation=aggregation,input_size=input_size,num_classes=num_classes)

class MyModel(nn.Module):
    def __init__(self,
                 input_size=224,
                 num_classes:int=3,
                 aggregation: str = 'Mean'
                 ):
        super(MyModel, self).__init__()

        self.aggregation = aggregation
        # self.vit = ViT(input_size=input_size,num_classes=num_classes, aggregation=aggregation)
        self.fpn = resnet_14_cat_swin_transformer_fpn(res_pretrain_path='resNet14.pth',
                                                      swin_pretrain_path='swin_my1.pth',aggregation=aggregation,input_size=input_size,num_classes=num_classes)
        self.lstm = nn.LSTM(input_size=input_size*input_size, hidden_size=512, num_layers=3)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x1,x2):
        if self.aggregation == 'LSTM':
            hidden = None
            for t in range(x1.size(1)):
                # with torch.no_grad():
                x = self.fpn(x1[:, t, :, :, :],x2[:, t, :, :, :])
                out, hidden = self.lstm(x.unsqueeze(0), hidden)

            x = self.fc1(out[-1, :, :])
            x = F.relu(x)
            x = self.fc2(x)
        else:
            B = x1.shape[0]
            x1_sequence= torch.flatten(x1, start_dim=0, end_dim=1)
            x2_sequence= torch.flatten(x2, start_dim=0, end_dim=1)
            x = self.fpn(x1_sequence,x2_sequence)
            x = x.view(B, int(x.shape[0] / B), x.shape[1])
            # x = torch.mean(x, 1)
            weights = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1]).to('cuda:0')
            # 使用 torch.einsum 进行加权平均计算
            x = torch.einsum('ijk,j->ik', x, weights)
            # fpn_out=self.fpn(x1,x2)
            # x=self.classify(fpn_out)

        return x


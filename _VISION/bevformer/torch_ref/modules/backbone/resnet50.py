import torch.nn as nn
import torchvision.models as models
from torch import Tensor as Tensor

class ResNet50(models.ResNet):
    def __init__(self, block=models.resnet.Bottleneck, layers = [3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet50, self).__init__(block, layers, num_classes, zero_init_residual,
                                       groups, width_per_group, replace_stride_with_dilation,
                                       norm_layer)
        # Define fc layer as Identity for not confusing when loading statedict
        self.fc = nn.Identity()
    
    
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Remove 2 last layers [avgpool, fc] 

        return x  
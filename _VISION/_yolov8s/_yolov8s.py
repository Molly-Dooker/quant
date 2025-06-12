import torch
import torch.nn as nn
import ipdb
class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    # default_act = nn.SiLU()  # default activation

    def __init__(self, module):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = module.conv
        # self.bn   = module.bn
        self.act  = module.act

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, module):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        # c_ = c1 // 2  # hidden channels
        # self.cv1 = Conv(c1, c_, 1, 1)
        # self.cv2 = Conv(c_ * 4, c2, 1, 1)
        # self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv1 = module.cv1
        self.cv2 = module.cv2
        self.m   = module.m

    def forward(self, x):
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, module):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = module.d

    def forward(self, x):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)

class C2f(torch.nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self,module):
        super().__init__()
        self.cv1 = module.cv1
        self.cv2 = module.cv2
        self.m   = module.m       

    def forward(self, x):
        """Forward pass through C2f layer."""
        x1, x2 = self.cv1(x).chunk(2, dim=1)
        outputs = [x1, x2]
        for block in self.m:
            x2 = block(x2)
            outputs.append(x2)
        result = self.cv2(torch.cat(outputs, dim=1))
        return result

class Detect(torch.nn.Module):
    def __init__(self, model, size=640):
        super().__init__()        
        self.nl = model.nl
        self.cv2 = model.cv2
        self.cv3 = model.cv3     
        self.dfl = model.dfl  
        self.register_buffer('stride', model.stride)
        self.no  = model.no
        self.reg_max = model.reg_max
        self.nc      = model.nc
        anchors, strides = self.make_anchors(size)
        self.register_buffer('anchors', anchors.contiguous())
        self.register_buffer('strides', strides.contiguous())

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        outs = []
        for i in range(self.nl):
            feat = x[i]
            outs.append(torch.cat((self.cv2[i](feat), self.cv3[i](feat)), dim=1))
        dbox = self._inference(outs)
        # return outs, dbox
        return dbox
    def _inference(self, x):
        out0 = x[0].flatten(2)
        out1 = x[1].flatten(2)
        out2 = x[2].flatten(2)
        x_cat = torch.cat((out0, out1, out2), dim=2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        distance = self.dfl(box)        
        dbox = self.dist2bbox(distance, self.anchors.unsqueeze(0))*self.strides
        result = torch.cat((dbox, cls.sigmoid()), 1)
        return result

    def make_anchors(self, size):
        """Generate anchors from features."""
        grid_cell_offset=0.5
        anchor_points, stride_tensor = [], []
        # assert feats is not None
        device = self.stride.device
        for i, stride in enumerate(self.stride):
            h = int(size/stride.item())
            w = int(size/stride.item())
            sx = torch.arange(end=w, device=device, ) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, ) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, device=device))
        return torch.cat(anchor_points).transpose(0, 1), torch.cat(stride_tensor).transpose(0, 1)

    def dist2bbox(self, distance, anchor_points):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = distance.chunk(2, 1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        # if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), 1)

class Yolov8s(torch.nn.Module):
    def __init__(self, model, size=640):
        super().__init__()
        self.m0  = Conv(model[0])
        self.m1  = Conv(model[1])
        self.m2  = C2f(model[2])
        self.m3  = Conv(model[3])
        self.m4  = C2f(model[4])
        self.m5  = Conv(model[5])
        self.m6  = C2f(model[6])
        self.m7  = Conv(model[7])
        self.m8  = C2f(model[8])
        self.m9  = SPPF(model[9])
        self.m10 = torch.nn.Upsample(scale_factor=model[10].scale_factor, mode=model[10].mode)
        self.m11 = Concat(model[11])
        self.m12 = C2f(model[12])
        self.m13 = torch.nn.Upsample(scale_factor=model[13].scale_factor, mode=model[13].mode)
        self.m14 = Concat(model[14])
        self.m15 = C2f(model[15])
        self.m16 = Conv(model[16])
        self.m17 = Concat(model[17])
        self.m18 = C2f(model[18])
        self.m19 = Conv(model[19])
        self.m20 = Concat(model[20])
        self.m21 = C2f(model[21])
        self.m22 = Detect(model[22], size)

    def forward(self, x):       
        r0  = self.m0(x)    
        r1  = self.m1(r0)        
        r2  = self.m2(r1)     
        r3  = self.m3(r2)
        r4  = self.m4(r3)
        r5  = self.m5(r4)
        r6  = self.m6(r5)
        r7  = self.m7(r6)
        r8  = self.m8(r7)
        r9  = self.m9(r8)
        r10 = self.m10(r9)
        r11 = self.m11([r10,r6])
        r12 = self.m12(r11)        
        r13 = self.m13(r12)
        r14 = self.m14([r13,r4])
        r15 = self.m15(r14)
        r16 = self.m16(r15)
        r17 = self.m17([r16,r12])
        r18 = self.m18(r17)
        r19 = self.m19(r18)
        r20 = self.m20([r19,r9])
        r21 = self.m21(r20)
        r22 = self.m22([r15,r18,r21])        
        return r22
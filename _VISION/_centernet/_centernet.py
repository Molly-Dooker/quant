import torch
import cv2
from torch import nn
import numpy as np
import time
from os.path import join
import torch.utils.model_zoo as model_zoo
from torchvision.ops import deform_conv2d
from torch.nn.modules.utils import _pair
import math
import sys
import os
from pathlib import Path
from utils.debugger import Debugger
from utils.functions import get_affine_transform
from utils.functions import ctdet_post_process, flip_tensor, ctdet_decode
import ipdb



BN_MOMENTUM = 0.1
save_dir = Path(".")    #! REPLACE YOUR SAVE DIR
tensor_ext = ".pt"

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DCNv2Function(nn.Module):
    def __init__(self, stride=1, padding=0, dilation=1, deformable_groups=1):
        super(DCNv2Function, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

    def forward(self, input, offset, mask, weight, bias=None):
        # Use torchvision's deform_conv2d
        output = deform_conv2d(
            input,
            offset,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return output

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        func = DCNv2Function(self.stride, self.padding, self.dilation, self.deformable_groups)
        return func(input, offset, mask, self.weight, self.bias)


class DCN(DCNv2):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=(self.stride, self.stride),
                                          padding=(self.padding, self.padding),
                                          bias=True)
        self.init_offset()
        self.func = DCNv2Function(self.stride, self.padding, self.dilation, self.deformable_groups)

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        result = self.func(input, offset, mask, self.weight, self.bias)
        return result


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()        
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x
class deformconv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride:int,
                 padding:int,
                 dilation:int,
                 bias: bool = True,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3,3).to(**factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels).to(**factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation

    def forward(self, input, offset, mask):
        # Use torchvision's deform_conv2d
        output = deform_conv2d(
            input    = input,
            offset   = offset,
            weight   = self.weight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            mask     = mask,
        )
        return output
    def extra_repr(self):
      return f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}"


class DeformConv2(nn.Module):
    def __init__(self, module):
        super(DeformConv2, self).__init__()
        self.conv_offset_mask = module.conv.conv_offset_mask
        ########################################### batchnorm folding
        self.deformconv2d = deformconv2d(
          in_channels = module.conv.weight.shape[1],
          out_channels = module.conv.weight.shape[0],
          stride = module.conv.stride,
          padding = module.conv.padding,
          dilation= module.conv.dilation)
        self.deformconv2d.weight.data.copy_(module.conv.weight)
        if module.conv.bias is not None:
          self.deformconv2d.bias.data.copy_(module.conv.bias)
        bn = module.actf[0]
        W = self.deformconv2d.weight  # [out_c, in_c, k, k]
        b = self.deformconv2d.bias
        if b is None:
            b = torch.zeros(W.size(0), device=W.device)
        gamma = bn.weight       # [out_c]
        beta  = bn.bias         # [out_c]
        mu    = bn.running_mean # [out_c]
        var   = bn.running_var  # [out_c]
        eps   = bn.eps
        # 3) scale 계수 α 계산
        alpha = gamma / torch.sqrt(var + eps)  # [out_c]
        # 4) folded weight/bias 계산
        #   - weight는 각 아웃채널마다 α를 곱해줍니다.
        W_fold = W * alpha.view(-1, 1, 1, 1)
        b_fold = beta - alpha * mu + alpha * b
        # 5) deform_conv 모듈에 반영
        self.deformconv2d.weight.data.copy_(W_fold)
        self.deformconv2d.bias = torch.nn.Parameter(b_fold)
        self.bn = torch.nn.Identity()
        ########################################### batchnorm folding

        self.relu = module.actf[1]
        del module
    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        result = self.deformconv2d(input, offset, mask)        
        x = self.bn(result)
        x = self.relu(x)
        return x



class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f): # channels = 
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])

class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
            
        x = self.relu(x)

        return x

class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        print("level: ", levels)
        print("channels: ", channels)
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)


    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86', model_dir = None):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(url=model_url, model_dir=model_dir)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)



class DLASeg(nn.Module):
    def __init__(self, base_name = "dla34", heads = {'hm': 80, 'wh': 2, 'reg': 2}, weightpath=None ,pretrained = True, down_ratio = 4, final_kernel =1,
                 last_level = 5 , head_conv = 256, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock)
        self.base.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86', model_dir = weightpath)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        # y = []
        # for i in range(self.last_level - self.first_level): # last_level = 5, first_level = 2
        #     y.append(x[i].clone())
        y = x[:3]
        # self.ida_up(y, 0, len(y))
        self.ida_up(y,0,3)
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        # return [z]
        return z
    



def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
	start_epoch = 0
	checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
	print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
	state_dict_ = checkpoint['state_dict']
	state_dict = {}
	# convert data_parallal to model
	for k in state_dict_:
		if k.startswith('module') and not k.startswith('module_list'):
			state_dict[k[7:]] = state_dict_[k]
		else:
			state_dict[k] = state_dict_[k]
	model_state_dict = model.state_dict()

	# check loaded parameters and created model parameters
	msg = 'If you see this, your model does not fully load the ' + \
		'pre-trained weight. Please make sure ' + \
		'you have correctly specified --arch xxx ' + \
		'or set the correct --num_classes for your own dataset.'
	for k in state_dict:
		if k in model_state_dict:
			if state_dict[k].shape != model_state_dict[k].shape:
				print('Skip loading parameter {}, required shape{}, '\
				'loaded shape{}. {}'.format(k, model_state_dict[k].shape, state_dict[k].shape, msg))
				state_dict[k] = model_state_dict[k]
		else:
			print('Drop parameter {}.'.format(k) + msg)
	for k in model_state_dict:
		if not (k in state_dict):
			print('No param {}.'.format(k) + msg)
			state_dict[k] = model_state_dict[k]
	model.load_state_dict(state_dict, strict=False)

	# resume optimizer parameters
	if optimizer is not None and resume:
		if 'optimizer' in checkpoint:
			optimizer.load_state_dict(checkpoint['optimizer'])
			start_epoch = checkpoint['epoch']
			start_lr = lr
			for step in lr_step:
				if start_epoch >= step:
					start_lr *= 0.1
			for param_group in optimizer.param_groups:
				param_group['lr'] = start_lr
			print('Resumed optimizer with start lr', start_lr)
		else:
			print('No optimizer parameters in checkpoint.')
	if optimizer is not None:
		return model, optimizer, start_epoch
	else:
		return model

class CenterNet(nn.Module):
	def __init__(self, opt):
		super(CenterNet, self).__init__()
		self.weight_path = opt.load_model
		print('Creating model...')
		print("opt: ", opt)
		self.mean  = opt.mean
		self.std   = opt.std
		self.model = DLASeg(pretrained=True, weightpath = os.path.dirname(self.weight_path)) # base weight 로드함. 이 weight 는 torch model zoo 에서 받아옴
		self.model = load_model(self.model, self.weight_path) # 나머지 weight 로드함. 이 weight 는 사전에 받아놓은 weight
		# self.model = self.model.to(device)
		self.model.eval()
		self.mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
		self.std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
		self.max_per_image = 100
		self.num_classes = opt.num_classes
		self.scales = opt.test_scales
		self.opt = opt
		self.pause = True
		# self.opt.device = device

	def pre_process(self, image, scale, meta=None):
		height, width = image.shape[0:2]
		new_height = int(height * scale)
		new_width  = int(width * scale)
		if self.opt.fix_res:
			inp_height, inp_width = self.opt.input_h, self.opt.input_w
			c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
			s = max(height, width) * 1.0
		else:
			inp_height = (new_height | self.opt.pad) + 1
			inp_width = (new_width | self.opt.pad) + 1
			c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
			s = np.array([inp_width, inp_height], dtype=np.float32)

		trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
		resized_image = cv2.resize(image, (new_width, new_height))
		inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
		inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

		images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
		if self.opt.flip_test:
			images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
		images = torch.from_numpy(images)
		meta = {'c': c, 's': s, 
				'out_height': inp_height // self.opt.down_ratio, 
				'out_width': inp_width // self.opt.down_ratio}
		return images, meta

	def process(self, images, return_time=False):
		with torch.no_grad():
			output = self.model(images)      
			hm = output['hm'].sigmoid_()
			wh = output['wh']
			reg = output['reg'] if self.opt.reg_offset else None
			if self.opt.flip_test:
				hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
				wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
				reg = reg[0:1] if reg is not None else None
			forward_time = time.time()       
			dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

		if return_time:
			return output, dets, forward_time
		else:
			return output, dets

	def post_process(self, dets, meta, scale=1):
		dets = dets.detach().cpu().numpy()
		dets = dets.reshape(1, -1, dets.shape[2])
		dets = ctdet_post_process(
      		dets.copy(), [meta['c']], [meta['s']],
			meta['out_height'], meta['out_width'], self.opt.num_classes)
		for j in range(1, self.num_classes + 1):
			dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
			dets[0][j][:, :4] /= scale
		return dets[0]

	def merge_outputs(self, detections):
		if isinstance(detections,dict): detections= [detections]    
		results = {}
		for j in range(1, self.num_classes + 1):
			results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)
		# if len(self.scales) > 1 or self.opt.nms:
		#    soft_nms(results[j], Nt=0.5, method=2)
		scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])
		if len(scores) > self.max_per_image:
			kth = len(scores) - self.max_per_image
			thresh = np.partition(scores, kth)[kth]
			for j in range(1, self.num_classes + 1):
				keep_inds = (results[j][:, 4] >= thresh)
				results[j] = results[j][keep_inds]
		return results

	def debug(self, debugger, images, dets, output, scale=1):
		detection = dets.detach().cpu().numpy().copy()
		detection[:, :, :4] *= self.opt.down_ratio
		for i in range(1):
			img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
			img = ((img * self.std + self.mean) * 255).astype(np.uint8)
			pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
			debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
			debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
			for k in range(len(dets[i])):
				if detection[i, k, 4] > self.opt.center_thresh:
					debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
									detection[i, k, 4], 
									img_id='out_pred_{:.1f}'.format(scale))

	def show_results(self, debugger, image, results):
		debugger.add_img(image, img_id='ctdet')
		for j in range(1, self.num_classes + 1):
			for bbox in results[j]:
				if bbox[4] > self.opt.vis_thresh:
					debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
		debugger.save_img(imgId='ctdet', path='./')

	def run(self, image_or_path_or_tensor, meta=None):
		load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
		merge_time, tot_time = 0, 0
		debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                      theme=self.opt.debugger_theme)
		start_time = time.time()
		pre_processed = False
		if isinstance(image_or_path_or_tensor, np.ndarray):
			image = image_or_path_or_tensor
		elif type(image_or_path_or_tensor) == type (''): 
			image = cv2.imread(image_or_path_or_tensor)
		else:
			image = image_or_path_or_tensor['image'][0].numpy()
			pre_processed_images = image_or_path_or_tensor
			pre_processed = True
		
		loaded_time = time.time()
		load_time += (loaded_time - start_time)
		
		detections = []
		
		for scale in self.scales:
			scale_start_time = time.time()
			if not pre_processed: #pre_processed = False
				images, meta = self.pre_process(image, scale, meta)
			else:
				# import pdb; pdb.set_trace()
				images = pre_processed_images['images'][scale][0]
				meta = pre_processed_images['meta'][scale]
				meta = {k: v.numpy()[0] for k, v in meta.items()}
			images = images.to(self.opt.device)
			pre_process_time = time.time()
			pre_time += pre_process_time - scale_start_time

			output, dets, forward_time = self.process(images, return_time=True)

			net_time += forward_time - pre_process_time
			decode_time = time.time()
			dec_time += decode_time - forward_time
			# if self.opt.debug >= 2:
			#   self.debug(debugger, images, dets, output, scale)
			dets_ = self.post_process(dets, meta, scale)
			post_process_time = time.time()
			post_time += post_process_time - decode_time
			detections.append(dets_)

		results = self.merge_outputs(detections)

		end_time = time.time()
		merge_time += end_time - post_process_time
		tot_time += end_time - start_time
		# print("Debug")
		self.show_results(debugger, image, results)
		
		return {'results': results, 'tot': tot_time, 'load': load_time,
				'pre': pre_time, 'net': net_time, 'dec': dec_time,
				'post': post_time, 'merge': merge_time}



class Config: 
	def __init__(self,load_model, device):
		self.K=100
		self.aggr_weight=0.0
		self.agnostic_ex=False
		self.arch='dla_34'
		self.aug_ddd=0.5
		self.aug_rot=0
		self.batch_size=32
		self.cat_spec_wh=False
		self.center_thresh=0.1
		self.chunk_sizes=[32]
		self.dataset='coco'
		self.debug=0
		self.debugger_theme='white'
		self.demo='.'
		self.dense_hp=False
		self.dense_wh=False
		self.dep_weight=1
		self.dim_weight=1
		self.down_ratio=4
		self.eval_oracle_dep=False
		self.eval_oracle_hm=False
		self.eval_oracle_hmhp=False
		self.eval_oracle_hp_offset=False
		self.eval_oracle_kps=False
		self.eval_oracle_offset=False
		self.eval_oracle_wh=False
		self.exp_id='default'
		self.fix_res=True
		self.flip=0.5
		self.flip_test=False
		self.gpus=[0]
		self.gpus_str='0'
		self.head_conv=256, 
		self.heads={'hm': 80, 'wh': 2, 'reg': 2}
		self.hide_data_time=False
		self.hm_hp=True
		self.hm_hp_weight=1
		self.hm_weight=1
		self.hp_weight=1
		self.input_h=512
		self.input_res=512
		self.input_w=512
		self.keep_res=False
		self.kitti_split='3dop'
		# self.load_model='../weights/ctdet_coco_dla_2x.pth'
		self.load_model = load_model
		self.device = device
		self.lr=0.000125
		self.lr_step=[90, 120]
		self.master_batch_size=32
		self.mean=[0.408, 0.447, 0.47]
		self.metric='loss'
		self.mse_loss=False
		self.nms=False
		self.no_color_aug=False
		self.norm_wh=False
		self.not_cuda_benchmark=False
		self.not_hm_hp=False
		self.not_prefetch_test=False
		self.not_rand_crop=False
		self.not_reg_bbox=False
		self.not_reg_hp_offset=False
		self.not_reg_offset=False
		self.num_classes=80
		self.num_epochs=140
		self.num_iters=-1
		self.num_stacks=1
		self.num_workers=4
		self.off_weight=1
		self.output_h=128
		self.output_res=128
		self.output_w=128
		self.pad=31
		self.peak_thresh=0.2
		self.print_iter=0
		self.rect_mask=False
		self.reg_bbox=True
		self.reg_hp_offset=True
		self.reg_loss='l1'
		self.reg_offset=True
		self.resume=False
		self.rot_weight=1
		self.rotate=0
		self.save_all=False
		self.scale=0.4
		self.scores_thresh=0.1
		self.seed=317
		self.shift=0.1
		self.std=[0.289, 0.274, 0.278]
		self.task='ctdet'
		self.test=False
		self.test_scales=[1.0]
		self.trainval=False
		self.val_intervals=5
		self.vis_thresh=0.3
		self.wh_weight=0.1


if __name__ == '__main__':
	opt = Config()
	torch.manual_seed(0)
	device = 'cuda:2'
	Ctdet = CenterNet(opt,device)
	img = cv2.imread("im2.jpg")
	Ctdet.run(img)

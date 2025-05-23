from typing import Tuple, Any
import math
import ttnn
import torch
import torch.nn as nn
from bos_metal import op
from tt.utils import get_list_shape

class Conv2dSplit(op.Conv2d):
    sub_conv = 0
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...] = 1,
        padding: Tuple[int, ...] = 0,
        dilation: Tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        activation: str = "relu",
        config: op.Conv2dConfig = None,
        conv_op_cache: dict = {},        # shared cache for conv2d operations
        **kwargs: Any,
    ):
        super(Conv2dSplit, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            activation=activation,
            config=config,
            conv_op_cache=conv_op_cache,
            **kwargs,
        )
        self.batch_divisor = [1,2,3,6] # [1,2,3,6]  #hardcode for now

    def divide(self, x: ttnn.Tensor, dim: int = -1):    #type: ignore
        B, H, W, C = x.shape
        H_out, W_out = (H-self.kernel_size[0])//self.stride[0] + 1, (W-self.kernel_size[1])//self.stride[1] + 1
        # recalculate input shape and split
        split_tensors = []
        if dim==1:
            carry = (H-self.kernel_size[0])%self.stride[0]
            if H_out%2==1:
                new_H_1 = H_out//2
                new_H_2 = H_out//2 + 1
            else:
                new_H_1=new_H_2=H_out//2

            new_H_1 = (new_H_1 - 1) * self.stride[0] + self.kernel_size[0]
            new_H_2 = (new_H_2 - 1) * self.stride[0] + self.kernel_size[0] + carry
            split_tensors.append(x[:, 0:new_H_1, :, :])
            split_tensors.append(x[:, H-new_H_2:H, :, :])
        else:
            carry = (W-self.kernel_size[1])%self.stride[1]
            if W_out%2==1:
                new_W_1 = W_out//2
                new_W_2 = W_out//2 + 1
            else:
                new_W_1=new_W_2=W_out//2
            
            new_W_1 = (new_W_1 - 1) * self.stride[1] + self.kernel_size[1]
            new_W_2 = (new_W_2 - 1) * self.stride[1] + self.kernel_size[1] + carry
            # Debugging for Enqueue Write Buffer
            split_tensors.append(x[:, :, 0:new_W_1, :])
            split_tensors.append(x[:, :, W-new_W_2:W, :])

        return split_tensors
    
    def strategy(self, x: ttnn.Tensor, threshold=3*450*400):    #type: ignore
        """
        x should be in shape of (B, H, W, C)
        """
        B, H, W, C = x.shape
        if B*H*W*C > threshold:
            if H > W:
                dim=1
            else:
                dim=2
            xs = self.divide(x, dim=dim)
            sub_x_0 = self.strategy(xs[0], threshold=threshold)
            # ttnn.deallocate(xs[0])
            sub_x_1 = self.strategy(xs[1], threshold=threshold)
            # ttnn.deallocate(xs[1])

            ret = ttnn.concat([sub_x_0, sub_x_1], dim)
            ret = ttnn.reallocate(ret)
            # ttnn.deallocate(sub_x_0)
            # ttnn.deallocate(sub_x_1)
            return ret
        else:
            input_shape = torch.tensor(self.input_shape)
            orig_input_shape = input_shape.clone()
            input_shape[0] = x.shape[0]
            input_shape[-1] = x.shape[-2]
            input_shape[-2] = x.shape[-3]

            self.set_shapes(input_shape.tolist())

            output_shape = torch.tensor(self.output_shape)
            output_shape = output_shape[[0, 2, 3, 1]]
            # print("Running forward unit conv, input shape: ", x.shape)
            # print("Running convolution {} times".format(Conv2dSplit.sub_conv))
            ret = super().forward(x)
            Conv2dSplit.sub_conv += 1

            self.set_shapes(orig_input_shape.tolist())

            ret = ttnn.to_layout(ret, ttnn.ROW_MAJOR_LAYOUT)
            ret = ttnn.reshape_on_device(ret, *output_shape.tolist())
            ret = ttnn.to_memory_config(ret, ttnn.L1_MEMORY_CONFIG)
            ret = ttnn.sharded_to_interleaved(ret, ttnn.L1_MEMORY_CONFIG)
            ret = ttnn.reallocate(ret)
            return ret

    def process_sub_batch(self, x: ttnn.Tensor, threshold:int=3*500*500, divisor:int = 3, split_padding = False):  #type: ignore
        split = 6
        divisor = min(divisor, x.shape[0])
        n_batch = x.shape[0]//divisor
        for i in range(divisor):
            x_i = x[i*n_batch:(i+1)*n_batch, :, :, :] if divisor != 1 else x
            x_i = x_i.row_major()
            x_i = ttnn.reallocate(x_i)
            x_i = ttnn.permute(x_i, (0, 3, 1, 2))  # (B, C, H, W)
            if self.padding[0] > 0 or self.padding[1] > 0:
                # x_i = ttnn.pad(x_i, ((self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), value=0)
                if split_padding:
                    num_splits = x_i.shape[1] // split
                    x_i_padded = []
                    for j in range(num_splits):
                        x_i_j = x_i[:, j*split : (j + 1)*split, :, :]
                        x_i_j = ttnn.reallocate(x_i_j)
                        x_i_j_padded = ttnn.pad(x_i_j, ((self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), value=0.0, use_multicore=True)
                        # x_i_j = ttnn.pad(x_i, ((self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), value=0)
                        x_i_padded.append(x_i_j_padded)
                # x_i = ttnn.pad(x_i.cpu(), ((self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), value=0.0).to(self.device)
                # x_i = ttnn.pad(x_i.cpu(), ((0, 0), (0, 0)), value=0.0).to(self.device)
                    x_i = ttnn.concat(x_i_padded, dim=1)
                    x_i = ttnn.reallocate(x_i)
                else:
                    x_i = ttnn.pad(x_i, ((self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), value=0.0)
                    # x_i = ttnn.pad(x_i.cpu(), ((self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), value=0.0).to(self.device)
                    # x_i = ttnn.pad(x_i.cpu(), ((0, 0), (0, 0)), value=0.0).to(self.device)
            x_i = ttnn.permute(x_i, (0, 2, 3, 1))  # (B, H, W, C)
            padding = self.padding
            self.padding = (0,0)

            x_i = self.strategy(x_i, threshold=threshold)
            self.padding = padding
            if i==0:
                out = x_i
            else:
                out = ttnn.concat([out, x_i], dim=0)
                # ttnn.deallocate(x_i)

        return out # ttnn.reallocate(out)

    def forward(self, x: ttnn.Tensor, threshold:int=3*600*500, divisor:int = 3, split_padding = False) -> ttnn.Tensor: #type: ignore
        B, H, W, C = x.shape
        # ori_input_shape = self.input_shape
        if self.padding[0] > 0 or self.padding[1] > 0:
            for divisor in self.batch_divisor:
                if B*H*W*C/divisor > 20*1.5*1024*1024/2:
                    continue
                else:
                    x = self.process_sub_batch(x, threshold=threshold, divisor=divisor, split_padding=split_padding)
                    # ttnn.reallocate(x)
                    # self.set_shapes(ori_input_shape)
                    return x

            raise ValueError("Input volume is too large for this operation")
        else:
            x = self.process_sub_batch(x, threshold=threshold, divisor=divisor, split_padding=split_padding)
            # ttnn.reallocate(x)
            # self.set_shapes(ori_input_shape)
            return x

class Conv2dSplitOutChannels(op.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sub_out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...] = 1,
        padding: Tuple[int, ...] = 0,
        dilation: Tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        activation: str = "relu",
        config: op.Conv2dConfig = None,
        **kwargs: Any,
    ):
        if out_channels % sub_out_channels != 0:
            raise ValueError("out_channels should be divisible by sub_out_channels")
        
        self.is_bias = bias
        super(Conv2dSplitOutChannels, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            activation=activation,
            config=config,
            conv_op_cache={},
            **kwargs,
        )
        self.sub_out_channels = min(sub_out_channels, out_channels)
        self.divisor = out_channels // self.sub_out_channels
        self.conv_op_caches = [{} for _ in range(self.divisor)]

        self.kwargs = kwargs
        self.flag = False

    def __make_layers(self):
        # self.out_channels = self.sub_out_channels
        self.layers = nn.ModuleList([
            Conv2dSplit(
                self.in_channels, 
                self.sub_out_channels, 
                self.kernel_size, 
                self.stride, 
                self.padding, 
                self.dilation, 
                self.groups, 
                self.is_bias, 
                activation=self.activation, 
                config=self.config,
                conv_op_cache=self.conv_op_caches[i], 
                **self.kwargs
            ) 
            for i in range(self.divisor)
        ])
        for i, layer in enumerate(self.layers):
            layer.config.deallocate_activation = self.config.deallocate_activation
            layer.weight = op.BaseParameter(
                self.weight[i*self.sub_out_channels:(i+1)*self.sub_out_channels, :, :, :], 
                device=self.device,
                map_func=self._prepare_weight
                )
            layer.input_shape = (1, self.in_channels, 1, self.in_channels)
            layer.output_shape = (1,self.sub_out_channels,1,self.sub_out_channels)
            if self.is_bias:
                layer.bias = op.BaseParameter(
                    self.bias[i*self.sub_out_channels:(i+1)*self.sub_out_channels], 
                    device=self.device,
                    map_func=self._prepare_weight
                    )

    def forward(self, x: ttnn.Tensor, threshold:int=3*500*500, divisor:int=6) -> ttnn.Tensor:   #type: ignore
        if not self.flag:
            self.__make_layers()
            self.flag=True

        for i, layer in enumerate(self.layers):
            x_i = layer(x, divisor=divisor, threshold=threshold)
            if i == 0:
                out = x_i
            else:
                out = ttnn.concat([out, x_i], 3)
                out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)
                # ttnn.deallocate(x_i)
                #TODO: not check yet
            # out = ttnn.reallocate(out)
        return out


class Conv2dSplitChannels(op.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        sub_in_channels: int = -1,
        sub_out_channels: int = -1,
        stride: Tuple[int, ...] = 1,
        padding: Tuple[int, ...] = 0,
        dilation: Tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        activation: str = "relu",
        config: op.Conv2dConfig = None,
        **kwargs: Any,
    ):
        super(Conv2dSplitChannels, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            activation=activation,
            config=config,
            conv_op_cache={},
            **kwargs,
        )
        self.is_bias = bias
        self.sub_in_channels = min(sub_in_channels, in_channels) if sub_in_channels > 0 else in_channels
        self.sub_out_channels = min(sub_out_channels, out_channels) if sub_out_channels > 0 else out_channels
        self.divisor = in_channels // self.sub_in_channels
        self.conv_op_caches = [{} for _ in range(self.divisor)]

        self.kwargs = kwargs
        self.flag = False
        self.adder = op.Add(deallocate_input=True) # inplace=True

    def __make_layers(self):
        self.layers = nn.ModuleList([
            Conv2dSplitOutChannels(
                self.sub_in_channels, 
                self.out_channels,
                self.sub_out_channels,
                self.kernel_size, 
                self.stride, 
                self.padding, 
                self.dilation, 
                self.groups, 
                self.is_bias if i==0 else False, 
                activation=self.activation, 
                config=self.config,
                **self.kwargs
            ) 
            for i in range(self.divisor)
        ])

        for i, layer in enumerate(self.layers):
            layer.config.deallocate_activation =False
            layer.weight = op.BaseParameter(
                self.weight[:, i*self.sub_in_channels:(i+1)*self.sub_in_channels, :, :], 
                device=self.device,
                map_func=self._prepare_weight
                )
            layer.input_shape = (1, self.sub_in_channels, 1, self.sub_in_channels)
            layer.output_shape = (1,self.out_channels,1,self.out_channels)
            if self.is_bias and i==0:
                layer.bias = self.bias

    def forward(self, x: ttnn.Tensor, threshold:int=3*500*500, divisor:int=6) -> ttnn.Tensor:   #type: ignore
        if not self.flag:
            self.__make_layers()
            self.flag=True

        for i, layer in enumerate(self.layers):
            x_i = layer(x[:, :, :, i*self.sub_in_channels:(i+1)*self.sub_in_channels], divisor=divisor, threshold=threshold)
            if i == 0:
                out = x_i
            else:
                out = self.adder([out, x_i])
                out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
                out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
                out = ttnn.reallocate(out)
                # ttnn.deallocate(x_i)
        return out

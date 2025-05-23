
import ttnn
import torch

def divide(x: ttnn.Tensor, op, dim: int = -1):  #type: ignore
    _, H, W, _ = x.shape
    H_out, W_out = (H-op.kernel_size[0])//op.stride[0] + 1, (W-op.kernel_size[1])//op.stride[1] + 1
    # recalculate input shape and split
    split_tensors = []
    if dim==1:
        carry = (H-op.kernel_size[0])%op.stride[0]
        if H_out%2==1:
            new_H_1 = H_out//2
            new_H_2 = H_out//2 + 1
        else:
            new_H_1=new_H_2=H_out//2

        new_H_1 = (new_H_1 - 1) * op.stride[0] + op.kernel_size[0]
        new_H_2 = (new_H_2 - 1) * op.stride[0] + op.kernel_size[0] + carry
        split_tensors.append(x[:, 0:new_H_1, :, :])
        split_tensors.append(x[:, H-new_H_2:H, :, :])
    else:
        carry = (W-op.kernel_size[1])%op.stride[1]
        if W_out%2==1:
            new_W_1 = W_out//2
            new_W_2 = W_out//2 + 1
        else:
            new_W_1=new_W_2=W_out//2
        
        new_W_1 = (new_W_1 - 1) * op.stride[1] + op.kernel_size[1]
        new_W_2 = (new_W_2 - 1) * op.stride[1] + op.kernel_size[1] + carry
        split_tensors.append(x[:, :, 0:new_W_1, :])
        split_tensors.append(x[:, :, W-new_W_2:W, :])
        
    ttnn.deallocate(x)
    return split_tensors

def conquer(x, op, threshold=3*500*500):
    B, H, W, C = x.shape
    if B*H*W*C > threshold:
        if H > W:
            dim=1
        else:
            dim=2
        xs = divide(x, op, dim=dim)
        sub_x_0 = conquer(xs[0], op, threshold=threshold)
        ttnn.deallocate(xs[0])
        sub_x_1 = conquer(xs[1], op, threshold=threshold)
        ttnn.deallocate(xs[1])
        sub_x_0 = ttnn.sharded_to_interleaved(sub_x_0, ttnn.L1_MEMORY_CONFIG)
        sub_x_1 = ttnn.sharded_to_interleaved(sub_x_1, ttnn.L1_MEMORY_CONFIG)

        ret = ttnn.concat([sub_x_0, sub_x_1], dim)
        ttnn.deallocate(sub_x_0)
        ttnn.deallocate(sub_x_1)
        return ret
    else:
        input_shape = torch.tensor(op.input_shape)
        orig_input_shape = input_shape.clone()
        input_shape[0] = x.shape[0]
        input_shape[-1] = x.shape[-2]
        input_shape[-2] = x.shape[-3]

        op.set_shapes(input_shape.tolist())

        output_shape = torch.tensor(op.output_shape)
        output_shape = output_shape[[0, 2, 3, 1]]   # (B, H, W, C)
        x = op(x)

        op.set_shapes(orig_input_shape.tolist())

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        return x

def run(x, op, threshold):
    B, H, W, C = x.shape
    if B*H*W*C > 20*1.5*1024*1024/2:
        for i in range(2):
            x1 = x[:, :, :, i*C//2:(i+1)*C//2]
            x1 = ttnn.permute(x1, (0, 3, 1, 2))  # (B, C, H, W)
            x1 = ttnn.pad(x1, ((op.padding[0], op.padding[0]), (op.padding[1], op.padding[1])), value=0)
            x1 = ttnn.permute(x1, (0, 2, 3, 1))  # (B, H, W, C)

            H, W = H + 2 * op.padding[0], W + 2 * op.padding[1]

            padding = op.padding
            op.padding = (0,0)
            op.set_shapes([B, C//2, H, W])

            x1 = conquer(x1, op, threshold=threshold)
            op.padding = padding

            if i==0:
                out = x1
            else:
                out = ttnn.concat([out, x1], dim=3)

        ttnn.deallocate(x)
        ttnn.deallocate(x1)
    else:
        out = conquer(x, op, threshold=threshold)
        ttnn.deallocate(x)
    return out
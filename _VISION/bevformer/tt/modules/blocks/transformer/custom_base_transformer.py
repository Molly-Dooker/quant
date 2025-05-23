import copy
from torch import nn as nn
from bos_metal.core import BaseModule
from bos_metal import device_box, op
from tt.modules.ops.ffn import FFN
from tt.modules.blocks.attn import (TemporalSelfAttention, SpatialCrossAttention, 
                                    MultiheadAttention, CustomMSDeformableAttention)

__all__ = ['MyCustomBaseTransformerLayer']

class MyCustomBaseTransformerLayer(BaseModule):
    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(MyCustomBaseTransformerLayer, self).__init__()

        self.batch_first = batch_first

        assert set(operation_order) & set(
            ['self_attn', 'norm', 'ffn', 'cross_attn']) == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = op.ModuleList()
        
        # TemporalSelfAttention
        attn_list = [attn_.pop('type') for attn_ in attn_cfgs]
        # for attn in attn_cfgs:
        #     attn['device'] = self.device
        attn_configs = {attn_list[i]: attn_cfgs[i] for i in range(len(attn_list))}
        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                if operation_name == 'self_attn':
                    if 'TemporalSelfAttention' in attn_configs.keys():        
                        attention = TemporalSelfAttention(**attn_configs['TemporalSelfAttention'])
                    elif 'MultiheadAttention' in attn_configs.keys(): 
                        attention = MultiheadAttention(**attn_configs['MultiheadAttention'])
                elif operation_name == 'cross_attn':
                    if 'SpatialCrossAttention' in attn_configs.keys():
                        attention = SpatialCrossAttention(**attn_configs['SpatialCrossAttention'])
                    elif 'CustomMSDeformableAttention' in attn_configs.keys():
                        attention = CustomMSDeformableAttention(**attn_configs['CustomMSDeformableAttention'])
                self.attentions.append(attention)
            
        self.embed_dims = self.attentions[0].embed_dims
        # if isinstance(ffn_cfgs, dict):
        #     ffn_cfgs.pop('type')
        # else:
        #     for ffn_ in ffn_cfgs:
        #         ffn_.pop('type')

        self.ffns = op.ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = dict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                FFN(**ffn_cfgs[ffn_index]))

        self.norms = op.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(op.LayerNorm(self.embed_dims))



if __name__ == '__main__':
    # Create configs
    attn_cfgs = [{'type': 'TemporalSelfAttention', 
                            'embed_dims': 256, 
                            'num_levels': 1}, 
             {'type': 'SpatialCrossAttention', 
                            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                            'deformable_attention': {'embed_dims': 256, 
                                                     'num_points': 8, 
                                                     'num_levels': 1}, 
                            'embed_dims': 256}]

    ffn_cfgs=dict(type='FFN',
                embed_dims=256,
                feedforward_channels=512,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),

    device = device_box.open({"device_id": 3})
    layer = MyCustomBaseTransformerLayer(attn_cfgs=attn_cfgs,
                                         ffn_cfgs=ffn_cfgs,
                                         operation_order=['self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'],
                                         norm_cfg=dict(type='LN'),
                                         batch_first=True)
    print(layer)
    device_box.close()
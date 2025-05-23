

from bos_metal import ttnn, op
import torch
from bevformer.utils import inverse_sigmoid
from tt.utils import from_torch_many
# from tt.modules.blocks.transformer import BaseTransformerLayer
from tt.modules.blocks.transformer.transformer_layer_sequence import TransformerLayerSequence

from bevformer.utils import assert_many

__all__ = ['DetrTransformerDecoder', 'DetectionTransformerDecoder']

class DetrTransformerDecoder(TransformerLayerSequence):
    count = 0
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(DetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            assert post_norm_cfg["type"] == "LN", "Only support LN for now. Now allow register new norm layer"
            # self.post_norm = build_norm_layer(post_norm_cfg,
                                            #   self.embed_dims)[1]
            self.post_norm = op.LayerNorm(self.embed_dims)
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        assert_many(query, f"DetrTransformerDecoder.input_query.{DetrTransformerDecoder.count}")

        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            assert_many(query, f"DetrTransformerDecoder.output_layer.{DetrTransformerDecoder.count}")
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
                    
        DetrTransformerDecoder.count += 1
        return ttnn.stack(intermediate)


class DetectionTransformerDecoder(TransformerLayerSequence):
    count = 0
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        
        assert_many(query, f"DetectionTransformerDecoder.input_query.{DetectionTransformerDecoder.count}")
        order = 0
        for lid, layer in enumerate(self.layers):
            
            self.cout(f"Running decoder layer {lid}")
            reference_points_input = reference_points[..., :2].unsqueeze(2)  
            # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs).squeeze(0)
            
            ## Debugging XTuan
            assert_many(output, f"DetectionTransformerDecoder.output_layer.{order}.{DetectionTransformerDecoder.count}")
            self.cout("Permute attention output")
            output = output.permute(1, 0, 2)
            # output = ttnn.reallocate(output)
            assert_many(output, f"DetectionTransformerDecoder.output_layer_permuted_f.{order}.{DetectionTransformerDecoder.count}")
            if reg_branches is not None:
                # refine the regression results
                self.cout(f"Refining regression results {lid}")
                tmp = reg_branches[lid](output).squeeze(0)
                assert_many(tmp, f"DetectionTransformerDecoder.tmp.{order}.{DetectionTransformerDecoder.count}")

                assert reference_points.shape[-1] == 3

                # TODO: 98% PCC at the end
                tmp = tmp.permute(2, 0, 1)
                # tmp = ttnn.reallocate(tmp)
                reference_points = reference_points.permute(2, 0, 1)
                # reference_points = ttnn.reallocate(reference_points)
                new_reference_points_01 = tmp[0:2] + ttnn.inverse_sigmoid(reference_points[0:2])
                # assert_many(new_reference_points_01.permute(1, 2, 0), f"DetectionTransformerDecoder.new_reference_points_01.{order}.{DetectionTransformerDecoder.count}")
                new_reference_points_2 = tmp[4:5] + ttnn.inverse_sigmoid(reference_points[2:3])
                # assert_many(new_reference_points_2.permute(1, 2, 0), f"DetectionTransformerDecoder.new_reference_points_02.{order}.{DetectionTransformerDecoder.count}")
                reference_points = ttnn.concat([new_reference_points_01, new_reference_points_2], dim=0)
                reference_points = reference_points.permute(1, 2, 0).sigmoid_()
                assert_many(reference_points, f"DetectionTransformerDecoder.new_reference_points.{order}.{DetectionTransformerDecoder.count}")
                
            output = output.permute(1, 0, 2)
            # output = ttnn.reallocate(output)
            assert_many(output, f"DetectionTransformerDecoder.output_layer_permuted_l.{order}.{DetectionTransformerDecoder.count}")
            if self.return_intermediate:
                self.cout("Appending intermediate output")
                #TODO: FALLBACK
                intermediate.append(output.torch())
                intermediate_reference_points.append(reference_points)
            order += 1

        if self.return_intermediate:
            self.cout("Returning intermediate output for Detection Decoder")
            # TODO: FALLBACK due to ttnn.stack LOW PCC
            return ttnn.from_torch_all(torch.stack(intermediate),
                                       dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT,
                                       device=self.device), \
                    ttnn.stack(intermediate_reference_points)

        self.cout("Returning output for Detection Decoder")
        
        assert_many(output, f"DetectionTransformerDecoder.output.{DetectionTransformerDecoder.count}")
        assert_many(reference_points, f"DetectionTransformerDecoder.reference_points.{DetectionTransformerDecoder.count}")
        
        DetectionTransformerDecoder.count += 1
        return output, reference_points



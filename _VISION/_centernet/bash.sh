#!/usr/bin/env bash

for i in {0..7}; do
  python main.py \
    --exclude 'ida_up.up_2, dla_up.ida_0.proj_1.deformconv2d, ida_up.node_2.deformconv2d, dla_up.ida_0.proj_1.conv_offset_mask, ida_up.node_1.deformconv2d, dla_up.ida_0.node_1.deformconv2d, dla_up.ida_0.up_1, base.base_layer.0, base.level0.0' \
    --prefix "ex10-$i" \
    --device "$i" &
done

wait
echo "모든 작업 완료!"
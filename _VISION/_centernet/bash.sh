#!/usr/bin/env bash

for i in {0..7}; do
  python main.py \
    --exclude 're:^dla_up.ida_0.*, re:^dla_up.ida_1.node_2.*, ida_up.up_1, re:^ida_up.node_1.*, ida_up.up_2, re:^ida_up.node_2.*' \
    --prefix "ex8-$i" \
    --device "$i" &
done

wait
echo "모든 작업 완료!"
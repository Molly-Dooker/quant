#!/bin/bash

# 디바이스 번호를 첫 번째 인자로 받음
DEVICE=$1

if [ -z "$DEVICE" ]; then
  echo "사용법: $0 <device_number>"
  exit 1
fi

# 시작 인덱스 = device * 10 (총 10개 prefix 사용)
START=$((DEVICE * 10))
# 종료 인덱스 = START + 8 (짝수만 반복하므로 5쌍 -> 0~8 짝수)
END=$((START + 8))

# 반복 실행 (짝수 인덱스만)
for i in $(seq $START 2 $END); do
  echo "Running detr.py with prefix _test$i on device $DEVICE..."
  python detr.py \
    --exclude 're:^model.backbone.conv_encoder.model.encoder.stages.0.layers.0.*, re:^model.encoder.layers.2.self_attn.*, re:^model.encoder.layers.5.self_attn.*' \
    --device "$DEVICE" \
    --prefix "_test$i"

  NEXT=$((i + 1))
  echo "Running detr.py with prefix _test$NEXT on device $DEVICE..."
  python detr.py \
    --exclude 're:^model.backbone.conv_encoder.model.encoder.stages.0.layers.0.*, re:^model.encoder.layers.2.*, re:^model.encoder.layers.5.*' \
    --device "$DEVICE" \
    --prefix "_test$NEXT"
done
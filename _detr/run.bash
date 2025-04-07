#!/bin/bash

for device in 4 5 6 7; do
    prefix="test$device"
    python detr.py --prefix "$prefix" --device "$device" &
done
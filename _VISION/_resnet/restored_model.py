


def forward(self, x : torch.Tensor) -> torch.Tensor:
    conv1_input_scale_0 = self.conv1_input_scale_0
    conv1_input_zero_point_0 = self.conv1_input_zero_point_0
    quantize_per_tensor = torch.quantize_per_tensor(x, conv1_input_scale_0, conv1_input_zero_point_0, torch.quint8);  x = conv1_input_scale_0 = conv1_input_zero_point_0 = None
    conv1 = self.conv1(quantize_per_tensor);  quantize_per_tensor = None
    dequantize_1 = conv1.dequantize();  conv1 = None
    bn1 = self.bn1(dequantize_1);  dequantize_1 = None
    relu = self.relu(bn1);  bn1 = None
    maxpool = self.maxpool(relu);  relu = None
    layer1_0_conv1_input_scale_0 = self.layer1_0_conv1_input_scale_0
    layer1_0_conv1_input_zero_point_0 = self.layer1_0_conv1_input_zero_point_0
    quantize_per_tensor_2 = torch.quantize_per_tensor(maxpool, layer1_0_conv1_input_scale_0, layer1_0_conv1_input_zero_point_0, torch.quint8);  maxpool = layer1_0_conv1_input_scale_0 = layer1_0_conv1_input_zero_point_0 = None
    layer1_0_conv1 = getattr(self.layer1, "0").conv1(quantize_per_tensor_2)
    dequantize_3 = layer1_0_conv1.dequantize();  layer1_0_conv1 = None
    layer1_0_bn1 = getattr(self.layer1, "0").bn1(dequantize_3);  dequantize_3 = None
    layer1_0_relu = getattr(self.layer1, "0").relu(layer1_0_bn1);  layer1_0_bn1 = None
    layer1_0_conv2_input_scale_0 = self.layer1_0_conv2_input_scale_0
    layer1_0_conv2_input_zero_point_0 = self.layer1_0_conv2_input_zero_point_0
    quantize_per_tensor_4 = torch.quantize_per_tensor(layer1_0_relu, layer1_0_conv2_input_scale_0, layer1_0_conv2_input_zero_point_0, torch.quint8);  layer1_0_relu = layer1_0_conv2_input_scale_0 = layer1_0_conv2_input_zero_point_0 = None
    layer1_0_conv2 = getattr(self.layer1, "0").conv2(quantize_per_tensor_4);  quantize_per_tensor_4 = None
    dequantize_5 = layer1_0_conv2.dequantize();  layer1_0_conv2 = None
    layer1_0_bn2 = getattr(self.layer1, "0").bn2(dequantize_5);  dequantize_5 = None
    layer1_0_relu_1 = getattr(self.layer1, "0").relu(layer1_0_bn2);  layer1_0_bn2 = None
    layer1_0_conv3_input_scale_0 = self.layer1_0_conv3_input_scale_0
    layer1_0_conv3_input_zero_point_0 = self.layer1_0_conv3_input_zero_point_0
    quantize_per_tensor_6 = torch.quantize_per_tensor(layer1_0_relu_1, layer1_0_conv3_input_scale_0, layer1_0_conv3_input_zero_point_0, torch.quint8);  layer1_0_relu_1 = layer1_0_conv3_input_scale_0 = layer1_0_conv3_input_zero_point_0 = None
    layer1_0_conv3 = getattr(self.layer1, "0").conv3(quantize_per_tensor_6);  quantize_per_tensor_6 = None
    dequantize_7 = layer1_0_conv3.dequantize();  layer1_0_conv3 = None
    layer1_0_bn3 = getattr(self.layer1, "0").bn3(dequantize_7);  dequantize_7 = None
    layer1_0_downsample_0 = getattr(getattr(self.layer1, "0").downsample, "0")(quantize_per_tensor_2);  quantize_per_tensor_2 = None
    dequantize_8 = layer1_0_downsample_0.dequantize();  layer1_0_downsample_0 = None
    layer1_0_downsample_1 = getattr(getattr(self.layer1, "0").downsample, "1")(dequantize_8);  dequantize_8 = None
    add = layer1_0_bn3 + layer1_0_downsample_1;  layer1_0_bn3 = layer1_0_downsample_1 = None
    layer1_0_relu_2 = getattr(self.layer1, "0").relu(add);  add = None
    layer1_1_conv1_input_scale_0 = self.layer1_1_conv1_input_scale_0
    layer1_1_conv1_input_zero_point_0 = self.layer1_1_conv1_input_zero_point_0
    quantize_per_tensor_9 = torch.quantize_per_tensor(layer1_0_relu_2, layer1_1_conv1_input_scale_0, layer1_1_conv1_input_zero_point_0, torch.quint8);  layer1_1_conv1_input_scale_0 = layer1_1_conv1_input_zero_point_0 = None
    layer1_1_conv1 = getattr(self.layer1, "1").conv1(quantize_per_tensor_9);  quantize_per_tensor_9 = None
    dequantize_10 = layer1_1_conv1.dequantize();  layer1_1_conv1 = None
    layer1_1_bn1 = getattr(self.layer1, "1").bn1(dequantize_10);  dequantize_10 = None
    layer1_1_relu = getattr(self.layer1, "1").relu(layer1_1_bn1);  layer1_1_bn1 = None
    layer1_1_conv2_input_scale_0 = self.layer1_1_conv2_input_scale_0
    layer1_1_conv2_input_zero_point_0 = self.layer1_1_conv2_input_zero_point_0
    quantize_per_tensor_11 = torch.quantize_per_tensor(layer1_1_relu, layer1_1_conv2_input_scale_0, layer1_1_conv2_input_zero_point_0, torch.quint8);  layer1_1_relu = layer1_1_conv2_input_scale_0 = layer1_1_conv2_input_zero_point_0 = None
    layer1_1_conv2 = getattr(self.layer1, "1").conv2(quantize_per_tensor_11);  quantize_per_tensor_11 = None
    dequantize_12 = layer1_1_conv2.dequantize();  layer1_1_conv2 = None
    layer1_1_bn2 = getattr(self.layer1, "1").bn2(dequantize_12);  dequantize_12 = None
    layer1_1_relu_1 = getattr(self.layer1, "1").relu(layer1_1_bn2);  layer1_1_bn2 = None
    layer1_1_conv3_input_scale_0 = self.layer1_1_conv3_input_scale_0
    layer1_1_conv3_input_zero_point_0 = self.layer1_1_conv3_input_zero_point_0
    quantize_per_tensor_13 = torch.quantize_per_tensor(layer1_1_relu_1, layer1_1_conv3_input_scale_0, layer1_1_conv3_input_zero_point_0, torch.quint8);  layer1_1_relu_1 = layer1_1_conv3_input_scale_0 = layer1_1_conv3_input_zero_point_0 = None
    layer1_1_conv3 = getattr(self.layer1, "1").conv3(quantize_per_tensor_13);  quantize_per_tensor_13 = None
    dequantize_14 = layer1_1_conv3.dequantize();  layer1_1_conv3 = None
    layer1_1_bn3 = getattr(self.layer1, "1").bn3(dequantize_14);  dequantize_14 = None
    add_1 = layer1_1_bn3 + layer1_0_relu_2;  layer1_1_bn3 = layer1_0_relu_2 = None
    layer1_1_relu_2 = getattr(self.layer1, "1").relu(add_1);  add_1 = None
    layer1_2_conv1_input_scale_0 = self.layer1_2_conv1_input_scale_0
    layer1_2_conv1_input_zero_point_0 = self.layer1_2_conv1_input_zero_point_0
    quantize_per_tensor_15 = torch.quantize_per_tensor(layer1_1_relu_2, layer1_2_conv1_input_scale_0, layer1_2_conv1_input_zero_point_0, torch.quint8);  layer1_2_conv1_input_scale_0 = layer1_2_conv1_input_zero_point_0 = None
    layer1_2_conv1 = getattr(self.layer1, "2").conv1(quantize_per_tensor_15);  quantize_per_tensor_15 = None
    dequantize_16 = layer1_2_conv1.dequantize();  layer1_2_conv1 = None
    layer1_2_bn1 = getattr(self.layer1, "2").bn1(dequantize_16);  dequantize_16 = None
    layer1_2_relu = getattr(self.layer1, "2").relu(layer1_2_bn1);  layer1_2_bn1 = None
    layer1_2_conv2_input_scale_0 = self.layer1_2_conv2_input_scale_0
    layer1_2_conv2_input_zero_point_0 = self.layer1_2_conv2_input_zero_point_0
    quantize_per_tensor_17 = torch.quantize_per_tensor(layer1_2_relu, layer1_2_conv2_input_scale_0, layer1_2_conv2_input_zero_point_0, torch.quint8);  layer1_2_relu = layer1_2_conv2_input_scale_0 = layer1_2_conv2_input_zero_point_0 = None
    layer1_2_conv2 = getattr(self.layer1, "2").conv2(quantize_per_tensor_17);  quantize_per_tensor_17 = None
    dequantize_18 = layer1_2_conv2.dequantize();  layer1_2_conv2 = None
    layer1_2_bn2 = getattr(self.layer1, "2").bn2(dequantize_18);  dequantize_18 = None
    layer1_2_relu_1 = getattr(self.layer1, "2").relu(layer1_2_bn2);  layer1_2_bn2 = None
    layer1_2_conv3_input_scale_0 = self.layer1_2_conv3_input_scale_0
    layer1_2_conv3_input_zero_point_0 = self.layer1_2_conv3_input_zero_point_0
    quantize_per_tensor_19 = torch.quantize_per_tensor(layer1_2_relu_1, layer1_2_conv3_input_scale_0, layer1_2_conv3_input_zero_point_0, torch.quint8);  layer1_2_relu_1 = layer1_2_conv3_input_scale_0 = layer1_2_conv3_input_zero_point_0 = None
    layer1_2_conv3 = getattr(self.layer1, "2").conv3(quantize_per_tensor_19);  quantize_per_tensor_19 = None
    dequantize_20 = layer1_2_conv3.dequantize();  layer1_2_conv3 = None
    layer1_2_bn3 = getattr(self.layer1, "2").bn3(dequantize_20);  dequantize_20 = None
    add_2 = layer1_2_bn3 + layer1_1_relu_2;  layer1_2_bn3 = layer1_1_relu_2 = None
    layer1_2_relu_2 = getattr(self.layer1, "2").relu(add_2);  add_2 = None
    layer2_0_conv1_input_scale_0 = self.layer2_0_conv1_input_scale_0
    layer2_0_conv1_input_zero_point_0 = self.layer2_0_conv1_input_zero_point_0
    quantize_per_tensor_21 = torch.quantize_per_tensor(layer1_2_relu_2, layer2_0_conv1_input_scale_0, layer2_0_conv1_input_zero_point_0, torch.quint8);  layer1_2_relu_2 = layer2_0_conv1_input_scale_0 = layer2_0_conv1_input_zero_point_0 = None
    layer2_0_conv1 = getattr(self.layer2, "0").conv1(quantize_per_tensor_21)
    dequantize_22 = layer2_0_conv1.dequantize();  layer2_0_conv1 = None
    layer2_0_bn1 = getattr(self.layer2, "0").bn1(dequantize_22);  dequantize_22 = None
    layer2_0_relu = getattr(self.layer2, "0").relu(layer2_0_bn1);  layer2_0_bn1 = None
    layer2_0_conv2_input_scale_0 = self.layer2_0_conv2_input_scale_0
    layer2_0_conv2_input_zero_point_0 = self.layer2_0_conv2_input_zero_point_0
    quantize_per_tensor_23 = torch.quantize_per_tensor(layer2_0_relu, layer2_0_conv2_input_scale_0, layer2_0_conv2_input_zero_point_0, torch.quint8);  layer2_0_relu = layer2_0_conv2_input_scale_0 = layer2_0_conv2_input_zero_point_0 = None
    layer2_0_conv2 = getattr(self.layer2, "0").conv2(quantize_per_tensor_23);  quantize_per_tensor_23 = None
    dequantize_24 = layer2_0_conv2.dequantize();  layer2_0_conv2 = None
    layer2_0_bn2 = getattr(self.layer2, "0").bn2(dequantize_24);  dequantize_24 = None
    layer2_0_relu_1 = getattr(self.layer2, "0").relu(layer2_0_bn2);  layer2_0_bn2 = None
    layer2_0_conv3_input_scale_0 = self.layer2_0_conv3_input_scale_0
    layer2_0_conv3_input_zero_point_0 = self.layer2_0_conv3_input_zero_point_0
    quantize_per_tensor_25 = torch.quantize_per_tensor(layer2_0_relu_1, layer2_0_conv3_input_scale_0, layer2_0_conv3_input_zero_point_0, torch.quint8);  layer2_0_relu_1 = layer2_0_conv3_input_scale_0 = layer2_0_conv3_input_zero_point_0 = None
    layer2_0_conv3 = getattr(self.layer2, "0").conv3(quantize_per_tensor_25);  quantize_per_tensor_25 = None
    dequantize_26 = layer2_0_conv3.dequantize();  layer2_0_conv3 = None
    layer2_0_bn3 = getattr(self.layer2, "0").bn3(dequantize_26);  dequantize_26 = None
    layer2_0_downsample_0 = getattr(getattr(self.layer2, "0").downsample, "0")(quantize_per_tensor_21);  quantize_per_tensor_21 = None
    dequantize_27 = layer2_0_downsample_0.dequantize();  layer2_0_downsample_0 = None
    layer2_0_downsample_1 = getattr(getattr(self.layer2, "0").downsample, "1")(dequantize_27);  dequantize_27 = None
    add_3 = layer2_0_bn3 + layer2_0_downsample_1;  layer2_0_bn3 = layer2_0_downsample_1 = None
    layer2_0_relu_2 = getattr(self.layer2, "0").relu(add_3);  add_3 = None
    layer2_1_conv1_input_scale_0 = self.layer2_1_conv1_input_scale_0
    layer2_1_conv1_input_zero_point_0 = self.layer2_1_conv1_input_zero_point_0
    quantize_per_tensor_28 = torch.quantize_per_tensor(layer2_0_relu_2, layer2_1_conv1_input_scale_0, layer2_1_conv1_input_zero_point_0, torch.quint8);  layer2_1_conv1_input_scale_0 = layer2_1_conv1_input_zero_point_0 = None
    layer2_1_conv1 = getattr(self.layer2, "1").conv1(quantize_per_tensor_28);  quantize_per_tensor_28 = None
    dequantize_29 = layer2_1_conv1.dequantize();  layer2_1_conv1 = None
    layer2_1_bn1 = getattr(self.layer2, "1").bn1(dequantize_29);  dequantize_29 = None
    layer2_1_relu = getattr(self.layer2, "1").relu(layer2_1_bn1);  layer2_1_bn1 = None
    layer2_1_conv2_input_scale_0 = self.layer2_1_conv2_input_scale_0
    layer2_1_conv2_input_zero_point_0 = self.layer2_1_conv2_input_zero_point_0
    quantize_per_tensor_30 = torch.quantize_per_tensor(layer2_1_relu, layer2_1_conv2_input_scale_0, layer2_1_conv2_input_zero_point_0, torch.quint8);  layer2_1_relu = layer2_1_conv2_input_scale_0 = layer2_1_conv2_input_zero_point_0 = None
    layer2_1_conv2 = getattr(self.layer2, "1").conv2(quantize_per_tensor_30);  quantize_per_tensor_30 = None
    dequantize_31 = layer2_1_conv2.dequantize();  layer2_1_conv2 = None
    layer2_1_bn2 = getattr(self.layer2, "1").bn2(dequantize_31);  dequantize_31 = None
    layer2_1_relu_1 = getattr(self.layer2, "1").relu(layer2_1_bn2);  layer2_1_bn2 = None
    layer2_1_conv3_input_scale_0 = self.layer2_1_conv3_input_scale_0
    layer2_1_conv3_input_zero_point_0 = self.layer2_1_conv3_input_zero_point_0
    quantize_per_tensor_32 = torch.quantize_per_tensor(layer2_1_relu_1, layer2_1_conv3_input_scale_0, layer2_1_conv3_input_zero_point_0, torch.quint8);  layer2_1_relu_1 = layer2_1_conv3_input_scale_0 = layer2_1_conv3_input_zero_point_0 = None
    layer2_1_conv3 = getattr(self.layer2, "1").conv3(quantize_per_tensor_32);  quantize_per_tensor_32 = None
    dequantize_33 = layer2_1_conv3.dequantize();  layer2_1_conv3 = None
    layer2_1_bn3 = getattr(self.layer2, "1").bn3(dequantize_33);  dequantize_33 = None
    add_4 = layer2_1_bn3 + layer2_0_relu_2;  layer2_1_bn3 = layer2_0_relu_2 = None
    layer2_1_relu_2 = getattr(self.layer2, "1").relu(add_4);  add_4 = None
    layer2_2_conv1_input_scale_0 = self.layer2_2_conv1_input_scale_0
    layer2_2_conv1_input_zero_point_0 = self.layer2_2_conv1_input_zero_point_0
    quantize_per_tensor_34 = torch.quantize_per_tensor(layer2_1_relu_2, layer2_2_conv1_input_scale_0, layer2_2_conv1_input_zero_point_0, torch.quint8);  layer2_2_conv1_input_scale_0 = layer2_2_conv1_input_zero_point_0 = None
    layer2_2_conv1 = getattr(self.layer2, "2").conv1(quantize_per_tensor_34);  quantize_per_tensor_34 = None
    dequantize_35 = layer2_2_conv1.dequantize();  layer2_2_conv1 = None
    layer2_2_bn1 = getattr(self.layer2, "2").bn1(dequantize_35);  dequantize_35 = None
    layer2_2_relu = getattr(self.layer2, "2").relu(layer2_2_bn1);  layer2_2_bn1 = None
    layer2_2_conv2_input_scale_0 = self.layer2_2_conv2_input_scale_0
    layer2_2_conv2_input_zero_point_0 = self.layer2_2_conv2_input_zero_point_0
    quantize_per_tensor_36 = torch.quantize_per_tensor(layer2_2_relu, layer2_2_conv2_input_scale_0, layer2_2_conv2_input_zero_point_0, torch.quint8);  layer2_2_relu = layer2_2_conv2_input_scale_0 = layer2_2_conv2_input_zero_point_0 = None
    layer2_2_conv2 = getattr(self.layer2, "2").conv2(quantize_per_tensor_36);  quantize_per_tensor_36 = None
    dequantize_37 = layer2_2_conv2.dequantize();  layer2_2_conv2 = None
    layer2_2_bn2 = getattr(self.layer2, "2").bn2(dequantize_37);  dequantize_37 = None
    layer2_2_relu_1 = getattr(self.layer2, "2").relu(layer2_2_bn2);  layer2_2_bn2 = None
    layer2_2_conv3_input_scale_0 = self.layer2_2_conv3_input_scale_0
    layer2_2_conv3_input_zero_point_0 = self.layer2_2_conv3_input_zero_point_0
    quantize_per_tensor_38 = torch.quantize_per_tensor(layer2_2_relu_1, layer2_2_conv3_input_scale_0, layer2_2_conv3_input_zero_point_0, torch.quint8);  layer2_2_relu_1 = layer2_2_conv3_input_scale_0 = layer2_2_conv3_input_zero_point_0 = None
    layer2_2_conv3 = getattr(self.layer2, "2").conv3(quantize_per_tensor_38);  quantize_per_tensor_38 = None
    dequantize_39 = layer2_2_conv3.dequantize();  layer2_2_conv3 = None
    layer2_2_bn3 = getattr(self.layer2, "2").bn3(dequantize_39);  dequantize_39 = None
    add_5 = layer2_2_bn3 + layer2_1_relu_2;  layer2_2_bn3 = layer2_1_relu_2 = None
    layer2_2_relu_2 = getattr(self.layer2, "2").relu(add_5);  add_5 = None
    layer2_3_conv1_input_scale_0 = self.layer2_3_conv1_input_scale_0
    layer2_3_conv1_input_zero_point_0 = self.layer2_3_conv1_input_zero_point_0
    quantize_per_tensor_40 = torch.quantize_per_tensor(layer2_2_relu_2, layer2_3_conv1_input_scale_0, layer2_3_conv1_input_zero_point_0, torch.quint8);  layer2_3_conv1_input_scale_0 = layer2_3_conv1_input_zero_point_0 = None
    layer2_3_conv1 = getattr(self.layer2, "3").conv1(quantize_per_tensor_40);  quantize_per_tensor_40 = None
    dequantize_41 = layer2_3_conv1.dequantize();  layer2_3_conv1 = None
    layer2_3_bn1 = getattr(self.layer2, "3").bn1(dequantize_41);  dequantize_41 = None
    layer2_3_relu = getattr(self.layer2, "3").relu(layer2_3_bn1);  layer2_3_bn1 = None
    layer2_3_conv2_input_scale_0 = self.layer2_3_conv2_input_scale_0
    layer2_3_conv2_input_zero_point_0 = self.layer2_3_conv2_input_zero_point_0
    quantize_per_tensor_42 = torch.quantize_per_tensor(layer2_3_relu, layer2_3_conv2_input_scale_0, layer2_3_conv2_input_zero_point_0, torch.quint8);  layer2_3_relu = layer2_3_conv2_input_scale_0 = layer2_3_conv2_input_zero_point_0 = None
    layer2_3_conv2 = getattr(self.layer2, "3").conv2(quantize_per_tensor_42);  quantize_per_tensor_42 = None
    dequantize_43 = layer2_3_conv2.dequantize();  layer2_3_conv2 = None
    layer2_3_bn2 = getattr(self.layer2, "3").bn2(dequantize_43);  dequantize_43 = None
    layer2_3_relu_1 = getattr(self.layer2, "3").relu(layer2_3_bn2);  layer2_3_bn2 = None
    layer2_3_conv3_input_scale_0 = self.layer2_3_conv3_input_scale_0
    layer2_3_conv3_input_zero_point_0 = self.layer2_3_conv3_input_zero_point_0
    quantize_per_tensor_44 = torch.quantize_per_tensor(layer2_3_relu_1, layer2_3_conv3_input_scale_0, layer2_3_conv3_input_zero_point_0, torch.quint8);  layer2_3_relu_1 = layer2_3_conv3_input_scale_0 = layer2_3_conv3_input_zero_point_0 = None
    layer2_3_conv3 = getattr(self.layer2, "3").conv3(quantize_per_tensor_44);  quantize_per_tensor_44 = None
    dequantize_45 = layer2_3_conv3.dequantize();  layer2_3_conv3 = None
    layer2_3_bn3 = getattr(self.layer2, "3").bn3(dequantize_45);  dequantize_45 = None
    add_6 = layer2_3_bn3 + layer2_2_relu_2;  layer2_3_bn3 = layer2_2_relu_2 = None
    layer2_3_relu_2 = getattr(self.layer2, "3").relu(add_6);  add_6 = None
    layer3_0_conv1_input_scale_0 = self.layer3_0_conv1_input_scale_0
    layer3_0_conv1_input_zero_point_0 = self.layer3_0_conv1_input_zero_point_0
    quantize_per_tensor_46 = torch.quantize_per_tensor(layer2_3_relu_2, layer3_0_conv1_input_scale_0, layer3_0_conv1_input_zero_point_0, torch.quint8);  layer2_3_relu_2 = layer3_0_conv1_input_scale_0 = layer3_0_conv1_input_zero_point_0 = None
    layer3_0_conv1 = getattr(self.layer3, "0").conv1(quantize_per_tensor_46)
    dequantize_47 = layer3_0_conv1.dequantize();  layer3_0_conv1 = None
    layer3_0_bn1 = getattr(self.layer3, "0").bn1(dequantize_47);  dequantize_47 = None
    layer3_0_relu = getattr(self.layer3, "0").relu(layer3_0_bn1);  layer3_0_bn1 = None
    layer3_0_conv2_input_scale_0 = self.layer3_0_conv2_input_scale_0
    layer3_0_conv2_input_zero_point_0 = self.layer3_0_conv2_input_zero_point_0
    quantize_per_tensor_48 = torch.quantize_per_tensor(layer3_0_relu, layer3_0_conv2_input_scale_0, layer3_0_conv2_input_zero_point_0, torch.quint8);  layer3_0_relu = layer3_0_conv2_input_scale_0 = layer3_0_conv2_input_zero_point_0 = None
    layer3_0_conv2 = getattr(self.layer3, "0").conv2(quantize_per_tensor_48);  quantize_per_tensor_48 = None
    dequantize_49 = layer3_0_conv2.dequantize();  layer3_0_conv2 = None
    layer3_0_bn2 = getattr(self.layer3, "0").bn2(dequantize_49);  dequantize_49 = None
    layer3_0_relu_1 = getattr(self.layer3, "0").relu(layer3_0_bn2);  layer3_0_bn2 = None
    layer3_0_conv3_input_scale_0 = self.layer3_0_conv3_input_scale_0
    layer3_0_conv3_input_zero_point_0 = self.layer3_0_conv3_input_zero_point_0
    quantize_per_tensor_50 = torch.quantize_per_tensor(layer3_0_relu_1, layer3_0_conv3_input_scale_0, layer3_0_conv3_input_zero_point_0, torch.quint8);  layer3_0_relu_1 = layer3_0_conv3_input_scale_0 = layer3_0_conv3_input_zero_point_0 = None
    layer3_0_conv3 = getattr(self.layer3, "0").conv3(quantize_per_tensor_50);  quantize_per_tensor_50 = None
    dequantize_51 = layer3_0_conv3.dequantize();  layer3_0_conv3 = None
    layer3_0_bn3 = getattr(self.layer3, "0").bn3(dequantize_51);  dequantize_51 = None
    layer3_0_downsample_0 = getattr(getattr(self.layer3, "0").downsample, "0")(quantize_per_tensor_46);  quantize_per_tensor_46 = None
    dequantize_52 = layer3_0_downsample_0.dequantize();  layer3_0_downsample_0 = None
    layer3_0_downsample_1 = getattr(getattr(self.layer3, "0").downsample, "1")(dequantize_52);  dequantize_52 = None
    add_7 = layer3_0_bn3 + layer3_0_downsample_1;  layer3_0_bn3 = layer3_0_downsample_1 = None
    layer3_0_relu_2 = getattr(self.layer3, "0").relu(add_7);  add_7 = None
    layer3_1_conv1_input_scale_0 = self.layer3_1_conv1_input_scale_0
    layer3_1_conv1_input_zero_point_0 = self.layer3_1_conv1_input_zero_point_0
    quantize_per_tensor_53 = torch.quantize_per_tensor(layer3_0_relu_2, layer3_1_conv1_input_scale_0, layer3_1_conv1_input_zero_point_0, torch.quint8);  layer3_1_conv1_input_scale_0 = layer3_1_conv1_input_zero_point_0 = None
    layer3_1_conv1 = getattr(self.layer3, "1").conv1(quantize_per_tensor_53);  quantize_per_tensor_53 = None
    dequantize_54 = layer3_1_conv1.dequantize();  layer3_1_conv1 = None
    layer3_1_bn1 = getattr(self.layer3, "1").bn1(dequantize_54);  dequantize_54 = None
    layer3_1_relu = getattr(self.layer3, "1").relu(layer3_1_bn1);  layer3_1_bn1 = None
    layer3_1_conv2_input_scale_0 = self.layer3_1_conv2_input_scale_0
    layer3_1_conv2_input_zero_point_0 = self.layer3_1_conv2_input_zero_point_0
    quantize_per_tensor_55 = torch.quantize_per_tensor(layer3_1_relu, layer3_1_conv2_input_scale_0, layer3_1_conv2_input_zero_point_0, torch.quint8);  layer3_1_relu = layer3_1_conv2_input_scale_0 = layer3_1_conv2_input_zero_point_0 = None
    layer3_1_conv2 = getattr(self.layer3, "1").conv2(quantize_per_tensor_55);  quantize_per_tensor_55 = None
    dequantize_56 = layer3_1_conv2.dequantize();  layer3_1_conv2 = None
    layer3_1_bn2 = getattr(self.layer3, "1").bn2(dequantize_56);  dequantize_56 = None
    layer3_1_relu_1 = getattr(self.layer3, "1").relu(layer3_1_bn2);  layer3_1_bn2 = None
    layer3_1_conv3_input_scale_0 = self.layer3_1_conv3_input_scale_0
    layer3_1_conv3_input_zero_point_0 = self.layer3_1_conv3_input_zero_point_0
    quantize_per_tensor_57 = torch.quantize_per_tensor(layer3_1_relu_1, layer3_1_conv3_input_scale_0, layer3_1_conv3_input_zero_point_0, torch.quint8);  layer3_1_relu_1 = layer3_1_conv3_input_scale_0 = layer3_1_conv3_input_zero_point_0 = None
    layer3_1_conv3 = getattr(self.layer3, "1").conv3(quantize_per_tensor_57);  quantize_per_tensor_57 = None
    dequantize_58 = layer3_1_conv3.dequantize();  layer3_1_conv3 = None
    layer3_1_bn3 = getattr(self.layer3, "1").bn3(dequantize_58);  dequantize_58 = None
    add_8 = layer3_1_bn3 + layer3_0_relu_2;  layer3_1_bn3 = layer3_0_relu_2 = None
    layer3_1_relu_2 = getattr(self.layer3, "1").relu(add_8);  add_8 = None
    layer3_2_conv1_input_scale_0 = self.layer3_2_conv1_input_scale_0
    layer3_2_conv1_input_zero_point_0 = self.layer3_2_conv1_input_zero_point_0
    quantize_per_tensor_59 = torch.quantize_per_tensor(layer3_1_relu_2, layer3_2_conv1_input_scale_0, layer3_2_conv1_input_zero_point_0, torch.quint8);  layer3_2_conv1_input_scale_0 = layer3_2_conv1_input_zero_point_0 = None
    layer3_2_conv1 = getattr(self.layer3, "2").conv1(quantize_per_tensor_59);  quantize_per_tensor_59 = None
    dequantize_60 = layer3_2_conv1.dequantize();  layer3_2_conv1 = None
    layer3_2_bn1 = getattr(self.layer3, "2").bn1(dequantize_60);  dequantize_60 = None
    layer3_2_relu = getattr(self.layer3, "2").relu(layer3_2_bn1);  layer3_2_bn1 = None
    layer3_2_conv2_input_scale_0 = self.layer3_2_conv2_input_scale_0
    layer3_2_conv2_input_zero_point_0 = self.layer3_2_conv2_input_zero_point_0
    quantize_per_tensor_61 = torch.quantize_per_tensor(layer3_2_relu, layer3_2_conv2_input_scale_0, layer3_2_conv2_input_zero_point_0, torch.quint8);  layer3_2_relu = layer3_2_conv2_input_scale_0 = layer3_2_conv2_input_zero_point_0 = None
    layer3_2_conv2 = getattr(self.layer3, "2").conv2(quantize_per_tensor_61);  quantize_per_tensor_61 = None
    dequantize_62 = layer3_2_conv2.dequantize();  layer3_2_conv2 = None
    layer3_2_bn2 = getattr(self.layer3, "2").bn2(dequantize_62);  dequantize_62 = None
    layer3_2_relu_1 = getattr(self.layer3, "2").relu(layer3_2_bn2);  layer3_2_bn2 = None
    layer3_2_conv3_input_scale_0 = self.layer3_2_conv3_input_scale_0
    layer3_2_conv3_input_zero_point_0 = self.layer3_2_conv3_input_zero_point_0
    quantize_per_tensor_63 = torch.quantize_per_tensor(layer3_2_relu_1, layer3_2_conv3_input_scale_0, layer3_2_conv3_input_zero_point_0, torch.quint8);  layer3_2_relu_1 = layer3_2_conv3_input_scale_0 = layer3_2_conv3_input_zero_point_0 = None
    layer3_2_conv3 = getattr(self.layer3, "2").conv3(quantize_per_tensor_63);  quantize_per_tensor_63 = None
    dequantize_64 = layer3_2_conv3.dequantize();  layer3_2_conv3 = None
    layer3_2_bn3 = getattr(self.layer3, "2").bn3(dequantize_64);  dequantize_64 = None
    add_9 = layer3_2_bn3 + layer3_1_relu_2;  layer3_2_bn3 = layer3_1_relu_2 = None
    layer3_2_relu_2 = getattr(self.layer3, "2").relu(add_9);  add_9 = None
    layer3_3_conv1_input_scale_0 = self.layer3_3_conv1_input_scale_0
    layer3_3_conv1_input_zero_point_0 = self.layer3_3_conv1_input_zero_point_0
    quantize_per_tensor_65 = torch.quantize_per_tensor(layer3_2_relu_2, layer3_3_conv1_input_scale_0, layer3_3_conv1_input_zero_point_0, torch.quint8);  layer3_3_conv1_input_scale_0 = layer3_3_conv1_input_zero_point_0 = None
    layer3_3_conv1 = getattr(self.layer3, "3").conv1(quantize_per_tensor_65);  quantize_per_tensor_65 = None
    dequantize_66 = layer3_3_conv1.dequantize();  layer3_3_conv1 = None
    layer3_3_bn1 = getattr(self.layer3, "3").bn1(dequantize_66);  dequantize_66 = None
    layer3_3_relu = getattr(self.layer3, "3").relu(layer3_3_bn1);  layer3_3_bn1 = None
    layer3_3_conv2_input_scale_0 = self.layer3_3_conv2_input_scale_0
    layer3_3_conv2_input_zero_point_0 = self.layer3_3_conv2_input_zero_point_0
    quantize_per_tensor_67 = torch.quantize_per_tensor(layer3_3_relu, layer3_3_conv2_input_scale_0, layer3_3_conv2_input_zero_point_0, torch.quint8);  layer3_3_relu = layer3_3_conv2_input_scale_0 = layer3_3_conv2_input_zero_point_0 = None
    layer3_3_conv2 = getattr(self.layer3, "3").conv2(quantize_per_tensor_67);  quantize_per_tensor_67 = None
    dequantize_68 = layer3_3_conv2.dequantize();  layer3_3_conv2 = None
    layer3_3_bn2 = getattr(self.layer3, "3").bn2(dequantize_68);  dequantize_68 = None
    layer3_3_relu_1 = getattr(self.layer3, "3").relu(layer3_3_bn2);  layer3_3_bn2 = None
    layer3_3_conv3_input_scale_0 = self.layer3_3_conv3_input_scale_0
    layer3_3_conv3_input_zero_point_0 = self.layer3_3_conv3_input_zero_point_0
    quantize_per_tensor_69 = torch.quantize_per_tensor(layer3_3_relu_1, layer3_3_conv3_input_scale_0, layer3_3_conv3_input_zero_point_0, torch.quint8);  layer3_3_relu_1 = layer3_3_conv3_input_scale_0 = layer3_3_conv3_input_zero_point_0 = None
    layer3_3_conv3 = getattr(self.layer3, "3").conv3(quantize_per_tensor_69);  quantize_per_tensor_69 = None
    dequantize_70 = layer3_3_conv3.dequantize();  layer3_3_conv3 = None
    layer3_3_bn3 = getattr(self.layer3, "3").bn3(dequantize_70);  dequantize_70 = None
    add_10 = layer3_3_bn3 + layer3_2_relu_2;  layer3_3_bn3 = layer3_2_relu_2 = None
    layer3_3_relu_2 = getattr(self.layer3, "3").relu(add_10);  add_10 = None
    layer3_4_conv1_input_scale_0 = self.layer3_4_conv1_input_scale_0
    layer3_4_conv1_input_zero_point_0 = self.layer3_4_conv1_input_zero_point_0
    quantize_per_tensor_71 = torch.quantize_per_tensor(layer3_3_relu_2, layer3_4_conv1_input_scale_0, layer3_4_conv1_input_zero_point_0, torch.quint8);  layer3_4_conv1_input_scale_0 = layer3_4_conv1_input_zero_point_0 = None
    layer3_4_conv1 = getattr(self.layer3, "4").conv1(quantize_per_tensor_71);  quantize_per_tensor_71 = None
    dequantize_72 = layer3_4_conv1.dequantize();  layer3_4_conv1 = None
    layer3_4_bn1 = getattr(self.layer3, "4").bn1(dequantize_72);  dequantize_72 = None
    layer3_4_relu = getattr(self.layer3, "4").relu(layer3_4_bn1);  layer3_4_bn1 = None
    layer3_4_conv2_input_scale_0 = self.layer3_4_conv2_input_scale_0
    layer3_4_conv2_input_zero_point_0 = self.layer3_4_conv2_input_zero_point_0
    quantize_per_tensor_73 = torch.quantize_per_tensor(layer3_4_relu, layer3_4_conv2_input_scale_0, layer3_4_conv2_input_zero_point_0, torch.quint8);  layer3_4_relu = layer3_4_conv2_input_scale_0 = layer3_4_conv2_input_zero_point_0 = None
    layer3_4_conv2 = getattr(self.layer3, "4").conv2(quantize_per_tensor_73);  quantize_per_tensor_73 = None
    dequantize_74 = layer3_4_conv2.dequantize();  layer3_4_conv2 = None
    layer3_4_bn2 = getattr(self.layer3, "4").bn2(dequantize_74);  dequantize_74 = None
    layer3_4_relu_1 = getattr(self.layer3, "4").relu(layer3_4_bn2);  layer3_4_bn2 = None
    layer3_4_conv3_input_scale_0 = self.layer3_4_conv3_input_scale_0
    layer3_4_conv3_input_zero_point_0 = self.layer3_4_conv3_input_zero_point_0
    quantize_per_tensor_75 = torch.quantize_per_tensor(layer3_4_relu_1, layer3_4_conv3_input_scale_0, layer3_4_conv3_input_zero_point_0, torch.quint8);  layer3_4_relu_1 = layer3_4_conv3_input_scale_0 = layer3_4_conv3_input_zero_point_0 = None
    layer3_4_conv3 = getattr(self.layer3, "4").conv3(quantize_per_tensor_75);  quantize_per_tensor_75 = None
    dequantize_76 = layer3_4_conv3.dequantize();  layer3_4_conv3 = None
    layer3_4_bn3 = getattr(self.layer3, "4").bn3(dequantize_76);  dequantize_76 = None
    add_11 = layer3_4_bn3 + layer3_3_relu_2;  layer3_4_bn3 = layer3_3_relu_2 = None
    layer3_4_relu_2 = getattr(self.layer3, "4").relu(add_11);  add_11 = None
    layer3_5_conv1_input_scale_0 = self.layer3_5_conv1_input_scale_0
    layer3_5_conv1_input_zero_point_0 = self.layer3_5_conv1_input_zero_point_0
    quantize_per_tensor_77 = torch.quantize_per_tensor(layer3_4_relu_2, layer3_5_conv1_input_scale_0, layer3_5_conv1_input_zero_point_0, torch.quint8);  layer3_5_conv1_input_scale_0 = layer3_5_conv1_input_zero_point_0 = None
    layer3_5_conv1 = getattr(self.layer3, "5").conv1(quantize_per_tensor_77);  quantize_per_tensor_77 = None
    dequantize_78 = layer3_5_conv1.dequantize();  layer3_5_conv1 = None
    layer3_5_bn1 = getattr(self.layer3, "5").bn1(dequantize_78);  dequantize_78 = None
    layer3_5_relu = getattr(self.layer3, "5").relu(layer3_5_bn1);  layer3_5_bn1 = None
    layer3_5_conv2_input_scale_0 = self.layer3_5_conv2_input_scale_0
    layer3_5_conv2_input_zero_point_0 = self.layer3_5_conv2_input_zero_point_0
    quantize_per_tensor_79 = torch.quantize_per_tensor(layer3_5_relu, layer3_5_conv2_input_scale_0, layer3_5_conv2_input_zero_point_0, torch.quint8);  layer3_5_relu = layer3_5_conv2_input_scale_0 = layer3_5_conv2_input_zero_point_0 = None
    layer3_5_conv2 = getattr(self.layer3, "5").conv2(quantize_per_tensor_79);  quantize_per_tensor_79 = None
    dequantize_80 = layer3_5_conv2.dequantize();  layer3_5_conv2 = None
    layer3_5_bn2 = getattr(self.layer3, "5").bn2(dequantize_80);  dequantize_80 = None
    layer3_5_relu_1 = getattr(self.layer3, "5").relu(layer3_5_bn2);  layer3_5_bn2 = None
    layer3_5_conv3_input_scale_0 = self.layer3_5_conv3_input_scale_0
    layer3_5_conv3_input_zero_point_0 = self.layer3_5_conv3_input_zero_point_0
    quantize_per_tensor_81 = torch.quantize_per_tensor(layer3_5_relu_1, layer3_5_conv3_input_scale_0, layer3_5_conv3_input_zero_point_0, torch.quint8);  layer3_5_relu_1 = layer3_5_conv3_input_scale_0 = layer3_5_conv3_input_zero_point_0 = None
    layer3_5_conv3 = getattr(self.layer3, "5").conv3(quantize_per_tensor_81);  quantize_per_tensor_81 = None
    dequantize_82 = layer3_5_conv3.dequantize();  layer3_5_conv3 = None
    layer3_5_bn3 = getattr(self.layer3, "5").bn3(dequantize_82);  dequantize_82 = None
    add_12 = layer3_5_bn3 + layer3_4_relu_2;  layer3_5_bn3 = layer3_4_relu_2 = None
    layer3_5_relu_2 = getattr(self.layer3, "5").relu(add_12);  add_12 = None
    layer4_0_conv1_input_scale_0 = self.layer4_0_conv1_input_scale_0
    layer4_0_conv1_input_zero_point_0 = self.layer4_0_conv1_input_zero_point_0
    quantize_per_tensor_83 = torch.quantize_per_tensor(layer3_5_relu_2, layer4_0_conv1_input_scale_0, layer4_0_conv1_input_zero_point_0, torch.quint8);  layer3_5_relu_2 = layer4_0_conv1_input_scale_0 = layer4_0_conv1_input_zero_point_0 = None
    layer4_0_conv1 = getattr(self.layer4, "0").conv1(quantize_per_tensor_83)
    dequantize_84 = layer4_0_conv1.dequantize();  layer4_0_conv1 = None
    layer4_0_bn1 = getattr(self.layer4, "0").bn1(dequantize_84);  dequantize_84 = None
    layer4_0_relu = getattr(self.layer4, "0").relu(layer4_0_bn1);  layer4_0_bn1 = None
    layer4_0_conv2_input_scale_0 = self.layer4_0_conv2_input_scale_0
    layer4_0_conv2_input_zero_point_0 = self.layer4_0_conv2_input_zero_point_0
    quantize_per_tensor_85 = torch.quantize_per_tensor(layer4_0_relu, layer4_0_conv2_input_scale_0, layer4_0_conv2_input_zero_point_0, torch.quint8);  layer4_0_relu = layer4_0_conv2_input_scale_0 = layer4_0_conv2_input_zero_point_0 = None
    layer4_0_conv2 = getattr(self.layer4, "0").conv2(quantize_per_tensor_85);  quantize_per_tensor_85 = None
    dequantize_86 = layer4_0_conv2.dequantize();  layer4_0_conv2 = None
    layer4_0_bn2 = getattr(self.layer4, "0").bn2(dequantize_86);  dequantize_86 = None
    layer4_0_relu_1 = getattr(self.layer4, "0").relu(layer4_0_bn2);  layer4_0_bn2 = None
    layer4_0_conv3_input_scale_0 = self.layer4_0_conv3_input_scale_0
    layer4_0_conv3_input_zero_point_0 = self.layer4_0_conv3_input_zero_point_0
    quantize_per_tensor_87 = torch.quantize_per_tensor(layer4_0_relu_1, layer4_0_conv3_input_scale_0, layer4_0_conv3_input_zero_point_0, torch.quint8);  layer4_0_relu_1 = layer4_0_conv3_input_scale_0 = layer4_0_conv3_input_zero_point_0 = None
    layer4_0_conv3 = getattr(self.layer4, "0").conv3(quantize_per_tensor_87);  quantize_per_tensor_87 = None
    dequantize_88 = layer4_0_conv3.dequantize();  layer4_0_conv3 = None
    layer4_0_bn3 = getattr(self.layer4, "0").bn3(dequantize_88);  dequantize_88 = None
    layer4_0_downsample_0 = getattr(getattr(self.layer4, "0").downsample, "0")(quantize_per_tensor_83);  quantize_per_tensor_83 = None
    dequantize_89 = layer4_0_downsample_0.dequantize();  layer4_0_downsample_0 = None
    layer4_0_downsample_1 = getattr(getattr(self.layer4, "0").downsample, "1")(dequantize_89);  dequantize_89 = None
    add_13 = layer4_0_bn3 + layer4_0_downsample_1;  layer4_0_bn3 = layer4_0_downsample_1 = None
    layer4_0_relu_2 = getattr(self.layer4, "0").relu(add_13);  add_13 = None
    layer4_1_conv1_input_scale_0 = self.layer4_1_conv1_input_scale_0
    layer4_1_conv1_input_zero_point_0 = self.layer4_1_conv1_input_zero_point_0
    quantize_per_tensor_90 = torch.quantize_per_tensor(layer4_0_relu_2, layer4_1_conv1_input_scale_0, layer4_1_conv1_input_zero_point_0, torch.quint8);  layer4_1_conv1_input_scale_0 = layer4_1_conv1_input_zero_point_0 = None
    layer4_1_conv1 = getattr(self.layer4, "1").conv1(quantize_per_tensor_90);  quantize_per_tensor_90 = None
    dequantize_91 = layer4_1_conv1.dequantize();  layer4_1_conv1 = None
    layer4_1_bn1 = getattr(self.layer4, "1").bn1(dequantize_91);  dequantize_91 = None
    layer4_1_relu = getattr(self.layer4, "1").relu(layer4_1_bn1);  layer4_1_bn1 = None
    layer4_1_conv2_input_scale_0 = self.layer4_1_conv2_input_scale_0
    layer4_1_conv2_input_zero_point_0 = self.layer4_1_conv2_input_zero_point_0
    quantize_per_tensor_92 = torch.quantize_per_tensor(layer4_1_relu, layer4_1_conv2_input_scale_0, layer4_1_conv2_input_zero_point_0, torch.quint8);  layer4_1_relu = layer4_1_conv2_input_scale_0 = layer4_1_conv2_input_zero_point_0 = None
    layer4_1_conv2 = getattr(self.layer4, "1").conv2(quantize_per_tensor_92);  quantize_per_tensor_92 = None
    dequantize_93 = layer4_1_conv2.dequantize();  layer4_1_conv2 = None
    layer4_1_bn2 = getattr(self.layer4, "1").bn2(dequantize_93);  dequantize_93 = None
    layer4_1_relu_1 = getattr(self.layer4, "1").relu(layer4_1_bn2);  layer4_1_bn2 = None
    layer4_1_conv3_input_scale_0 = self.layer4_1_conv3_input_scale_0
    layer4_1_conv3_input_zero_point_0 = self.layer4_1_conv3_input_zero_point_0
    quantize_per_tensor_94 = torch.quantize_per_tensor(layer4_1_relu_1, layer4_1_conv3_input_scale_0, layer4_1_conv3_input_zero_point_0, torch.quint8);  layer4_1_relu_1 = layer4_1_conv3_input_scale_0 = layer4_1_conv3_input_zero_point_0 = None
    layer4_1_conv3 = getattr(self.layer4, "1").conv3(quantize_per_tensor_94);  quantize_per_tensor_94 = None
    dequantize_95 = layer4_1_conv3.dequantize();  layer4_1_conv3 = None
    layer4_1_bn3 = getattr(self.layer4, "1").bn3(dequantize_95);  dequantize_95 = None
    add_14 = layer4_1_bn3 + layer4_0_relu_2;  layer4_1_bn3 = layer4_0_relu_2 = None
    layer4_1_relu_2 = getattr(self.layer4, "1").relu(add_14);  add_14 = None
    layer4_2_conv1_input_scale_0 = self.layer4_2_conv1_input_scale_0
    layer4_2_conv1_input_zero_point_0 = self.layer4_2_conv1_input_zero_point_0
    quantize_per_tensor_96 = torch.quantize_per_tensor(layer4_1_relu_2, layer4_2_conv1_input_scale_0, layer4_2_conv1_input_zero_point_0, torch.quint8);  layer4_2_conv1_input_scale_0 = layer4_2_conv1_input_zero_point_0 = None
    layer4_2_conv1 = getattr(self.layer4, "2").conv1(quantize_per_tensor_96);  quantize_per_tensor_96 = None
    dequantize_97 = layer4_2_conv1.dequantize();  layer4_2_conv1 = None
    layer4_2_bn1 = getattr(self.layer4, "2").bn1(dequantize_97);  dequantize_97 = None
    layer4_2_relu = getattr(self.layer4, "2").relu(layer4_2_bn1);  layer4_2_bn1 = None
    layer4_2_conv2_input_scale_0 = self.layer4_2_conv2_input_scale_0
    layer4_2_conv2_input_zero_point_0 = self.layer4_2_conv2_input_zero_point_0
    quantize_per_tensor_98 = torch.quantize_per_tensor(layer4_2_relu, layer4_2_conv2_input_scale_0, layer4_2_conv2_input_zero_point_0, torch.quint8);  layer4_2_relu = layer4_2_conv2_input_scale_0 = layer4_2_conv2_input_zero_point_0 = None
    layer4_2_conv2 = getattr(self.layer4, "2").conv2(quantize_per_tensor_98);  quantize_per_tensor_98 = None
    dequantize_99 = layer4_2_conv2.dequantize();  layer4_2_conv2 = None
    layer4_2_bn2 = getattr(self.layer4, "2").bn2(dequantize_99);  dequantize_99 = None
    layer4_2_relu_1 = getattr(self.layer4, "2").relu(layer4_2_bn2);  layer4_2_bn2 = None
    layer4_2_conv3_input_scale_0 = self.layer4_2_conv3_input_scale_0
    layer4_2_conv3_input_zero_point_0 = self.layer4_2_conv3_input_zero_point_0
    quantize_per_tensor_100 = torch.quantize_per_tensor(layer4_2_relu_1, layer4_2_conv3_input_scale_0, layer4_2_conv3_input_zero_point_0, torch.quint8);  layer4_2_relu_1 = layer4_2_conv3_input_scale_0 = layer4_2_conv3_input_zero_point_0 = None
    layer4_2_conv3 = getattr(self.layer4, "2").conv3(quantize_per_tensor_100);  quantize_per_tensor_100 = None
    dequantize_101 = layer4_2_conv3.dequantize();  layer4_2_conv3 = None
    layer4_2_bn3 = getattr(self.layer4, "2").bn3(dequantize_101);  dequantize_101 = None
    add_15 = layer4_2_bn3 + layer4_1_relu_2;  layer4_2_bn3 = layer4_1_relu_2 = None
    layer4_2_relu_2 = getattr(self.layer4, "2").relu(add_15);  add_15 = None
    avgpool = self.avgpool(layer4_2_relu_2);  layer4_2_relu_2 = None
    flatten = torch.flatten(avgpool, 1);  avgpool = None
    fc_input_scale_0 = self.fc_input_scale_0
    fc_input_zero_point_0 = self.fc_input_zero_point_0
    quantize_per_tensor_102 = torch.quantize_per_tensor(flatten, fc_input_scale_0, fc_input_zero_point_0, torch.quint8);  flatten = fc_input_scale_0 = fc_input_zero_point_0 = None
    fc = self.fc(quantize_per_tensor_102);  quantize_per_tensor_102 = None
    dequantize_103 = fc.dequantize();  fc = None
    return dequantize_103
    
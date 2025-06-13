import onnx
import numpy as np
import json
import os
import argparse

def find_tensor_producer(graph, tensor_name):
    """주어진 이름의 텐서를 출력하는 노드를 찾습니다."""
    for node in graph.node:
        if tensor_name in node.output:
            return node
    return None

def get_param_shape(graph, param_name):
    """그래프의 Initializer에서 파라미터의 shape를 찾습니다."""
    for init in graph.initializer:
        if init.name == param_name:
            return init.dims
    return None


def create_qdq_onnx_model(
    original_model_path: str,
    encoding_path: str,
    output_model_path: str,
    root: str
):
    """
    (디버깅 버전) 원본 ONNX 모델과 인코딩 파일을 사용하여 수동으로 QDQ 형식의 ONNX 모델을 생성합니다.
    Per-channel quantization의 axis를 동적으로 결정하려고 시도합니다.
    """
    print(f"1. 로딩 및 전처리: '{original_model_path}' 와 '{encoding_path}'")
    model = onnx.load(original_model_path)
    graph = model.graph
    
    # ... (이전과 동일한 로딩 및 전처리 로직) ...
    model_output_names = {output.name for output in graph.output}
    with open(encoding_path, 'r') as f:
        encodings = json.load(f)
    # ...

    new_initializers = []
    qdq_info_map = {}
    initializer_names = {init.name for init in graph.initializer}
    all_encodings = encodings.get('param_encodings', []) + encodings.get('activation_encodings', [])

    for enc_info in all_encodings:
        tensor_name = enc_info['name']
        if tensor_name in model_output_names:
            print(f"   - 모델의 최종 출력 텐서 '{tensor_name}'는 양자화를 건너뜁니다.")
            continue
        
        # --- Axis 결정 로직 추가 ---
        axis = None
        scale_values = enc_info['scale']
        is_per_channel = isinstance(scale_values, list) and len(scale_values) > 1

        if is_per_channel:
            is_param = tensor_name in initializer_names
            if is_param:
                param_shape = get_param_shape(graph, tensor_name)
                # Conv, Gemm 가중치에 대한 일반적인 추론
                if len(param_shape) >= 2: # Conv (OC, IC, H, W) 또는 Gemm (OC, IC)
                    axis = 0 
                # 이 외의 경우는 더 복잡한 로직이 필요할 수 있음
            else: # Activation의 경우
                # ONNX에서 Activation의 per-channel 양자화는 보통 채널 차원인 1에 적용됨 (N, C, H, W)
                axis = 1 
            print(f"   - Per-channel 텐서 '{tensor_name}' (is_param={is_param})의 axis를 {axis}로 설정합니다.")
        # --- Axis 결정 로직 종료 ---


        # ... (이전과 동일한 스케일/ZP 텐서 생성 로직) ...
        offset_values = enc_info['offset']
        is_signed = enc_info['dtype'] == 'INT'
        onnx_dtype = onnx.TensorProto.INT8 if is_signed else onnx.TensorProto.UINT8
        numpy_dtype = np.int8 if is_signed else np.uint8
        scale_tensor_name = tensor_name + '_qdq_scale'
        zp_tensor_name = tensor_name + '_qdq_zero_point'
        scale_dims, zp_dims = [], []
        if is_per_channel:
            scale_dims, zp_dims = [len(scale_values)], [len(offset_values)]
        else:
            scale_values = [scale_values[0]] if isinstance(scale_values, list) else [scale_values]
            offset_values = [offset_values[0]] if isinstance(offset_values, list) else [offset_values]
        scale_tensor = onnx.helper.make_tensor(name=scale_tensor_name, data_type=onnx.TensorProto.FLOAT, dims=scale_dims, vals=np.array(scale_values, dtype=np.float32))
        zp_tensor = onnx.helper.make_tensor(name=zp_tensor_name, data_type=onnx_dtype, dims=zp_dims, vals=np.array(offset_values, dtype=numpy_dtype))
        new_initializers.extend([scale_tensor, zp_tensor])
        
        # ... (QDQ 노드 생성, axis 인자 사용) ...
        quantized_tensor_name = tensor_name + '_quantized'
        dequantized_tensor_name = tensor_name + '_dequantized'
        quant_node = onnx.helper.make_node('QuantizeLinear', inputs=[tensor_name, scale_tensor_name, zp_tensor_name], outputs=[quantized_tensor_name], name='QuantizeLinear_' + tensor_name.replace("::", "_"), axis=axis)
        dequant_node = onnx.helper.make_node('DequantizeLinear', inputs=[quantized_tensor_name, scale_tensor_name, zp_tensor_name], outputs=[dequantized_tensor_name], name='DequantizeLinear_' + tensor_name.replace("::", "_"), axis=axis)
        qdq_info_map[tensor_name] = {'new_name': dequantized_tensor_name, 'nodes': [quant_node, dequant_node], 'is_param': is_param}

    # ... (이후 그래프 재구성 및 저장 로직은 동일) ...
    print("\n2. 그래프를 재구성하며 QDQ 노드를 올바른 위치에 삽입합니다.")
    graph.initializer.extend(new_initializers)
    reordered_nodes = []
    original_nodes = list(graph.node)
    graph.ClearField('node')
    for node in original_nodes:
        for i, input_name in enumerate(node.input):
            if input_name in qdq_info_map:
                node.input[i] = qdq_info_map[input_name]['new_name']
    for inp in graph.input:
        if inp.name in qdq_info_map:
            reordered_nodes.extend(qdq_info_map[inp.name]['nodes'])
            print(f"  - 그래프 입력 '{inp.name}'에 대한 QDQ 노드 추가")
            qdq_info_map[inp.name]['inserted'] = True
    for node in original_nodes:
        for input_name in node.input:
            original_name = input_name.replace('_dequantized', '')
            if original_name in qdq_info_map and qdq_info_map[original_name]['is_param']:
                 if 'inserted' not in qdq_info_map[original_name]:
                    reordered_nodes.extend(qdq_info_map[original_name]['nodes'])
                    print(f"  - '{node.name}'가 사용하는 파라미터 '{original_name}'의 QDQ 노드 추가")
                    qdq_info_map[original_name]['inserted'] = True
    
        reordered_nodes.append(node)
    
        for output_name in node.output:
            if output_name in qdq_info_map:
                if 'inserted' not in qdq_info_map[output_name]:
                    reordered_nodes.extend(qdq_info_map[output_name]['nodes'])
                    print(f"  - '{node.name}'의 출력 '{output_name}' 뒤에 QDQ 노드 추가")
                    qdq_info_map[output_name]['inserted'] = True
    graph.node.extend(reordered_nodes)
    
    print("\n3. 수정된 모델을 검증하고 저장합니다.")
    try:
        onnx.checker.check_model(model)
        print("   - 모델 검증 성공!")
        onnx.save(model, output_model_path)
        graph_text_path = os.path.join(root, 'graph_qdq_structure.txt')
        with open(graph_text_path, "w") as f:
            for node in model.graph.node:
                f.write(str(node) + '\n')
        print(f"   - 최종 QDQ 모델이 '{output_model_path}'에 성공적으로 저장되었습니다.")
        print(f"   - 그래프 구조가 '{graph_text_path}'에 저장되었습니다.")
    except onnx.checker.ValidationError as e:
        print(f"   - 모델 검증 실패: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="수동으로 ONNX 모델에 QDQ 노드를 삽입합니다. (디버깅 버전)")
    parser.add_argument("--prefix", type=str, default="aa", help="결과 파일들이 저장된 폴더의 접두사")
    args = parser.parse_args()
    root = f'output/{args.prefix}/'
    if not os.path.exists(root):
        print(f"오류: '{root}' 경로를 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()
    ORIGINAL_MODEL_PATH = os.path.join(root, 'qq.onnx')
    ENCODING_PATH = os.path.join(root, 'qq.encodings')
    OUTPUT_QDQ_MODEL_PATH = os.path.join(root, 'qq_qdq.onnx')
    create_qdq_onnx_model(
        original_model_path=ORIGINAL_MODEL_PATH,
        encoding_path=ENCODING_PATH,
        output_model_path=OUTPUT_QDQ_MODEL_PATH,
        root=root
    )

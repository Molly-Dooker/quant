import onnx
import numpy as np
import json
import os
import argparse

def create_qdq_onnx_model(
    original_model_path: str,
    encoding_path: str,
    output_model_path: str,
    root: str
):
    """
    원본 ONNX 모델과 인코딩 파일을 사용하여 수동으로 QDQ 형식의 ONNX 모델을 생성합니다.
    이 함수는 그래프 위상 정렬을 고려하며, 모델의 최종 출력은 양자화하지 않습니다.

    :param original_model_path: 원본 FP32 ONNX 모델 파일 경로
    :param encoding_path: compute_encodings()로 생성된 encodings.json 파일 경로
    :param output_model_path: 최종 QDQ 모델이 저장될 경로
    :param root: 결과 파일(그래프 등)이 저장될 루트 폴더
    """
    # --- 1. 모델과 인코딩 정보 로드 및 전처리 ---
    print(f"1. 로딩 및 전처리: '{original_model_path}' 와 '{encoding_path}'")
    model = onnx.load(original_model_path)
    graph = model.graph

    # 모델의 최종 출력 텐서 이름들을 집합(set)으로 저장하여 빠른 조회를 위함
    model_output_names = {output.name for output in graph.output}
    print(f"   - 모델 최종 출력 텐서: {model_output_names}")

    # 인코딩 파일 전처리
    with open(encoding_path, 'r') as f:
        encodings = json.load(f)
    if 'version' not in encodings:
        encodings['version'] = '1.0.0'
    for enc_type in ['activation_encodings', 'param_encodings']:
        for enc in encodings.get(enc_type, []):
            if 'bw' in enc and 'bitwidth' not in enc:
                enc['bitwidth'] = enc.pop('bw')

    # --- 2. 그래프 수정을 위한 정보 수집 및 준비 ---
    new_initializers = []
    qdq_info_map = {} # 양자화할 텐서 정보를 모두 담을 맵
    
    # 모델의 가중치(initializer) 이름을 집합으로 만들어 빠른 조회를 위함
    initializer_names = {init.name for init in graph.initializer}

    all_encodings = encodings.get('param_encodings', []) + encodings.get('activation_encodings', [])
    
    for enc_info in all_encodings:
        tensor_name = enc_info['name']
        
        # *** 모델의 최종 출력 텐서는 양자화에서 제외 ***
        if tensor_name in model_output_names:
            print(f"   - 모델의 최종 출력 텐서 '{tensor_name}'는 양자화를 건너뜁니다.")
            continue

        is_param = tensor_name in initializer_names
        
        # 인코딩 정보 추출 및 스케일/ZP 텐서 생성
        scale_values = enc_info['scale']
        offset_values = enc_info['offset']
        is_signed = enc_info['dtype'] == 'INT'
        onnx_dtype = onnx.TensorProto.INT8 if is_signed else onnx.TensorProto.UINT8
        numpy_dtype = np.int8 if is_signed else np.uint8

        scale_tensor_name = tensor_name + '_qdq_scale'
        zp_tensor_name = tensor_name + '_qdq_zero_point'
        
        is_per_channel = isinstance(scale_values, list) and len(scale_values) > 1
        scale_dims, zp_dims, axis = [], [], None
        
        if is_per_channel:
            axis = 0
            scale_dims, zp_dims = [len(scale_values)], [len(offset_values)]
        else:
            scale_values = [scale_values[0]] if isinstance(scale_values, list) else [scale_values]
            offset_values = [offset_values[0]] if isinstance(offset_values, list) else [offset_values]

        scale_tensor = onnx.helper.make_tensor(name=scale_tensor_name, data_type=onnx.TensorProto.FLOAT, dims=scale_dims, vals=np.array(scale_values, dtype=np.float32))
        zp_tensor = onnx.helper.make_tensor(name=zp_tensor_name, data_type=onnx_dtype, dims=zp_dims, vals=np.array(offset_values, dtype=numpy_dtype))
        new_initializers.extend([scale_tensor, zp_tensor])
        
        # QDQ 노드 생성
        quantized_tensor_name = tensor_name + '_quantized'
        dequantized_tensor_name = tensor_name + '_dequantized'
        
        quant_node = onnx.helper.make_node('QuantizeLinear', inputs=[tensor_name, scale_tensor_name, zp_tensor_name], outputs=[quantized_tensor_name], name='QuantizeLinear_' + tensor_name.replace("::", "_"), axis=axis)
        dequant_node = onnx.helper.make_node('DequantizeLinear', inputs=[quantized_tensor_name, scale_tensor_name, zp_tensor_name], outputs=[dequantized_tensor_name], name='DequantizeLinear_' + tensor_name.replace("::", "_"), axis=axis)
        
        qdq_info_map[tensor_name] = {'new_name': dequantized_tensor_name, 'nodes': [quant_node, dequant_node], 'is_param': is_param}

    # --- 3. 그래프 재구성 ---
    print("\n2. 그래프를 재구성하며 QDQ 노드를 올바른 위치에 삽입합니다.")
    graph.initializer.extend(new_initializers)
    
    reordered_nodes = []
    original_nodes = list(graph.node)
    graph.ClearField('node')

    # 1단계: 원본 노드의 입력 연결을 QDQ 후의 이름으로 모두 변경
    for node in original_nodes:
        for i, input_name in enumerate(node.input):
            if input_name in qdq_info_map:
                node.input[i] = qdq_info_map[input_name]['new_name']

    # 2단계: 노드를 순서대로 다시 추가하면서 QDQ 노드를 삽입
    
    # 2-1: 모델 입력에 대한 QDQ 노드 먼저 추가
    for inp in graph.input:
        if inp.name in qdq_info_map:
            reordered_nodes.extend(qdq_info_map[inp.name]['nodes'])
            print(f"  - 그래프 입력 '{inp.name}'에 대한 QDQ 노드 추가")
            # 중복 추가 방지를 위해 처리 완료 표시
            qdq_info_map[inp.name]['inserted'] = True

    # 2-2: 기존 노드들을 순회하며 재구성
    for node in original_nodes:
        # 이 노드가 사용하는 가중치(파라미터)에 대한 QDQ 노드를 먼저 추가
        for input_name in node.input:
            original_name = input_name.replace('_dequantized', '')
            if original_name in qdq_info_map and qdq_info_map[original_name]['is_param']:
                 if 'inserted' not in qdq_info_map[original_name]:
                    reordered_nodes.extend(qdq_info_map[original_name]['nodes'])
                    print(f"  - '{node.name}'가 사용하는 파라미터 '{original_name}'의 QDQ 노드 추가")
                    qdq_info_map[original_name]['inserted'] = True

        # 원래 노드 추가
        reordered_nodes.append(node)

        # 이 노드가 생성하는 활성화에 대한 QDQ 노드를 바로 뒤에 추가
        for output_name in node.output:
            if output_name in qdq_info_map:
                if 'inserted' not in qdq_info_map[output_name]:
                    reordered_nodes.extend(qdq_info_map[output_name]['nodes'])
                    print(f"  - '{node.name}'의 출력 '{output_name}' 뒤에 QDQ 노드 추가")
                    qdq_info_map[output_name]['inserted'] = True
    
    graph.node.extend(reordered_nodes)

    # --- 4. 최종 모델 저장 ---
    print("\n3. 수정된 모델을 검증하고 저장합니다.")
    try:
        onnx.checker.check_model(model)
        print("   - 모델 검증 성공!")
        onnx.save(model, output_model_path)
        # 그래프 구조를 텍스트 파일로 저장 (디버깅용)
        graph_text_path = os.path.join(root, 'graph_qdq.graph')
        with open(graph_text_path, "w") as f:
            for node in model.graph.node:
                f.write(str(node) + '\n')
        print(f"   - 최종 QDQ 모델이 '{output_model_path}'에 성공적으로 저장되었습니다.")
        print(f"   - 그래프 구조가 '{graph_text_path}'에 저장되었습니다.")

    except onnx.checker.ValidationError as e:
        print(f"   - 모델 검증 실패: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="수동으로 ONNX 모델에 QDQ 노드를 삽입합니다.")
    parser.add_argument("--prefix", type=str, default="aa", help="결과 파일들이 저장된 폴더의 접두사")
    args = parser.parse_args()

    # --- 파일 경로 설정 ---
    root = f'output/{args.prefix}/'
    if not os.path.exists(root):
        print(f"오류: '{root}' 경로를 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()
        
    ORIGINAL_MODEL_PATH = os.path.join(root, 'qq.onnx')
    ENCODING_PATH = os.path.join(root, 'qq.encodings')
    OUTPUT_QDQ_MODEL_PATH = os.path.join(root, 'qq_qdq.onnx')
    
    # 스크립트 실행
    create_qdq_onnx_model(
        original_model_path=ORIGINAL_MODEL_PATH,
        encoding_path=ENCODING_PATH,
        output_model_path=OUTPUT_QDQ_MODEL_PATH,
        root=root
    )

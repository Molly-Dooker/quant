import compressed_tensors.linear
import compressed_tensors.linear.compressed_linear
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM   , PreTrainedTokenizerFast
import ipdb
import lm_eval
lm_eval.evaluator
# from llmcompressor.modifiers.quantization import QuantizationModifier
import compressed_tensors
compressed_tensors.linear.compressed_linear.CompressedLinear
import argparse
from compressed_tensors.quantization.lifecycle.forward import wrap_module_forward_quantized

def main(model_id):
    device = 'cuda:1'
    
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    def moderate(chat):

        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]
    
    out = moderate([
            messages
        ])
    print('-----')
    print(model_id)
    print(out)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, required=True, help='')

    # 커맨드라인 인자를 파싱합니다.
    args = parser.parse_args()    
    main(args.model)
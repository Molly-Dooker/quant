from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
import argparse
import compressed_tensors
compressed_tensors.quantization.quant_scheme
from compressed_tensors.quantization import QuantizationArgs;
from compressed_tensors.quantization.quant_args import QuantizationType, QuantizationStrategy
import ipdb, sys
from pathlib import Path
def main(model_id, saveroot, modelname, max_seq_len, num_samples, smoothing_strength, dampening_frac, gptq_ignore):
    # make savedir and set logger
    SAVE_DIR = Path(saveroot)/model_id.split("/")[-1]
    ignore_layers = ["lm_head"]
    if gptq_ignore is not None:
        modelname+='-gptqignore_'
        gptq_ignore = gptq_ignore.replace(' ','').split(',')
        for layer in gptq_ignore: 
            modelname+=layer+'_'
            ignore_layers.append(f"re:.*.{layer}")        
        modelname = modelname[:-1]
    SAVE_DIR= str(SAVE_DIR/modelname)
    logger.remove()
    logger.add(f"_logs/{SAVE_DIR}.log", rotation="500 MB", level="INFO")
    logger.add(sys.stderr, level="INFO", filter=lambda record: "hide_console" not in record["extra"])
    logger.info(f'{model_id} start')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    def preprocess_fn(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}
    ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
    ds = ds.shuffle(seed=24).select(range(num_samples))
    ds = ds.map(preprocess_fn)
    def tokenize(sample):
        return tokenizer(sample["text"], padding=False, max_length=max_seq_len, truncation=True, add_special_tokens=False)
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    recipe = [
        SmoothQuantModifier(
            smoothing_strength=smoothing_strength,
            mappings=[
            [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
            [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
            [["re:.*down_proj"], "re:.*up_proj"],
            ],
        ),
        GPTQModifier(
            sequential=True,
            targets="Linear",
            scheme="W8A8",
            ignore= ignore_layers,
            dampening_frac=dampening_frac,
        )
    ]
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=num_samples
        )
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    logger.info(f'{model_id} end')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, required=True, help='') # meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct
    parser.add_argument('--saveroot', type=str, default='_model/', help='')
    parser.add_argument('--modelname', type=str, default='quant', help='')
    parser.add_argument('--maxseq', type=int, default=4096, help='') # 4096
    parser.add_argument('--sample', type=int, default=512, help='') # 512
    parser.add_argument('--smoothing_strength', type=float, default=0.7, help='')
    parser.add_argument('--dampening_frac', type=float, default=0.01, help='')
    parser.add_argument('--gptq_ignore', type=str, required=False, help='')


    # 커맨드라인 인자를 파싱합니다.
    args = parser.parse_args()    
    main(args.model, args.saveroot, args.modelname, args.maxseq, args.sample, args.smoothing_strength, args.dampening_frac, args.gptq_ignore)









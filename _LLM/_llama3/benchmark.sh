accelerate launch --num_processes 8 -m lm_eval \
--model hf \
--model_args pretrained=./_model/Llama-3.2-1B-Instruct/quant,dtype=auto,add_bos_token=True,max_length=3850 \
--tasks mmlu_llama \
--batch_size 8  \
--fewshot_as_multiturn  \
--apply_chat_template \
--output_path ./_benchmark/ \
--num_fewshot 5


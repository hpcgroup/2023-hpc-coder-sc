import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorForSeq2Seq
from data_utils import alpaca_template

use_flash_attention = True
model_id = "/global/cfs/cdirs/m2404/ckpt/hpc-coder-v2-hf/"
model = AutoModelForCausalLM.from_pretrained(model_id,
                                                torch_dtype=torch.bfloat16,
                                                    attn_implementation='eager' if not use_flash_attention else "flash_attention_2").to('cuda')

model.eval()
template = alpaca_template["prompt_no_input"]

instructions = ['\nGiven a distributed file system where files are divided into blocks of equal size, design an algorithm to efficiently allocate and deallocate blocks across multiple storage nodes. Consider the following requirements:\n\n* Each block must be assigned to exactly one storage node.\n* The distribution of blocks should be balanced across the storage nodes to optimize performance.\n* The algorithm should minimize the number of block reassignments when a new block is allocated or an existing block is deallocated.\n',]

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")

with torch.no_grad():
    for instruction in instructions:
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            prompt = template.format(instruction=instruction)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            outputs = model.generate(input_ids.cuda(),
                                     do_sample=True, max_new_tokens=1024)

            print(tokenizer.batch_decode(outputs)[0])

import torch
import ray
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
# std imports
from collections import Counter
import json
import os
import time
from typing import Optional
# tpl imports
from alive_progress import alive_bar
import datasets

seed_dataset_id = 'hpcgroup/hpc-stack-seeds'
max_new_tokens = 2048
random_seed = 42
language_counter = Counter()
total_samples = 40000
prompt_template_1 = """You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions.
Please gain inspiration from the random code snippet below to create a high-quality programming problem. Be creative. Present your output in two distinct sections: **Problem Description** and **Solution**.
Provide code in the **Solution** section that solve the problem you describe in the **Problem Description** section.
You must use the following code snippet as inspiration for the problem you describe in the **Problem Description** section:
{seed}
Lines for each section of output:
1. **Problem Description**: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.
2. **Solution**: Offer a comprehensive, **correct**, **optimal** solution that accurately addresses the **Problem Description** you provided. The provided code should be **fast** and **efficient**."""

prompt_template_2 = """You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions.
Please gain inspiration from the following random code snippet to create a high-quality code optimization problem. Be creative.Present your output in two distinct sections: **Problem Description** and **Solution**.
Code snippet for inspiration:
{seed}
lines for each section:
1. **Problem Description**: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. The problem should require optimizing a piece of code. Provide this code to optimize in the problem description. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.
2. **Solution**: Offer a comprehensive, **correct** solution that accurately addresses the **Problem Description** you provided and optimizes the code. Include the optimized code."""

prompt_template_3 = """You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions.
Please gain inspiration from the following random code snippet to create a high-quality code translation problem. Present your output in two distinct sections: **Problem Description** and **Solution**.
Code snippet for inspiration:
{seed}
lines for each section:
1. **Problem Description**: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. The problem should require translating code between execution models (i.e. translating cuda to openmp or openmp to mpi or mpi to cuda or cuda to raja or raja to mpi or mpi to kokkos). Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.
2. **Solution**: Offer a comprehensive, **correct** solution that accurately addresses the **Problem Description** you provided and translates the code. Include the translated code."""

prompt_template_4 = """You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions.
Please gain inspiration from the following random code snippet to create a high-quality code parallelization problem.Present your output in two distinct sections: **Problem Description** and **Solution**.
Code snippet for inspiration:
{seed}
lines for each section:
1. **Problem Description**: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. The problem should require parallelizing a piece of code. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.
2. **Solution**: Offer a comprehensive, **correct** solution that accurately addresses the **Problem Description** you provided and parallelizes the code. Include the parallel code."""
results=[]
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
ray.init(num_gpus=2)
if ray.is_initialized():
    ray.shutdown()
llm = LLM(model="meta-llama/Meta-Llama-3-70B-Instruct",gpu_memory_utilization=0.95, max_num_seqs=1,swap_space=1,tensor_parallel_size=2)
print ("model loaded")
tokenizer = llm.get_tokenizer()
def postprocess(input_text: str) -> str:
    """ Postprocess the model output to return the text from each section.
        This is accomplished by finding lines that contain each section header.
    """
    lines = input_text.splitlines()
    problem_keyword = "**Problem Description**"
    solution_keyword = "**Solution**"
    if(input_text.find(problem_keyword) == -1):
        problem_keyword = "**Problem Description:**"
    if(input_text.find(solution_keyword) == -1):
        solution_keyword = "**Solution:**"

    if(input_text.find(problem_keyword) == -1 or input_text.find(solution_keyword) == -1):
        raise ValueError(f"All sections not present")

    # Find the starting index of each section
    problem_start = input_text.find(problem_keyword) + len(problem_keyword)
    solution_start = input_text.find(solution_keyword) + len(solution_keyword)

    # Extract the sections
    problem_description = input_text[problem_start:solution_start - len(solution_keyword)].strip()
    solution = input_text[solution_start:].strip()
    return problem_description, solution

def generate_output(prompts,total,sum) -> str:
    conversations = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate(conversations,
                       SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=4096,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        )
    )
    for output in outputs:
        time = (output.metrics.finished_time)-(output.metrics.first_token_time)
        total+=len(output.outputs[0].token_ids)/time
        sum.append(len(output.outputs[0].token_ids)/time)
        print(f"Tokens/second = ",len(output.outputs[0].token_ids)/time)
        return output.outputs[0].text

seed_dataset = datasets.load_dataset(seed_dataset_id, split='train', streaming=True).shuffle(seed=random_seed, buffer_size=50)
i = 0
total=0
sum=[]
with alive_bar(total_samples) as bar:
    bar(len(results), skipped=True)
    for element in seed_dataset:
      prompts=[]
      if(i > 57000):
        seed = element['text']
        try:
          if(i % 4 == 0):
            prompt = [{"role": "user", "content":prompt_template_4.format(seed=seed)}]
          elif(i % 3 == 0):
            prompt = [{"role": "user", "content":prompt_template_3.format(seed=seed)}]
          elif(i % 2 == 0):
            prompt = [{"role": "user", "content":prompt_template_2.format(seed=seed)}]
          else:
            prompt = [{"role": "user", "content":prompt_template_1.format(seed=seed)}]
          prompts.append(prompt)
          generated_text = generate_output(prompts,total,sum)
          problem_statement, solution = postprocess(generated_text)
        except Exception as e:
          print("Error:{e}")
          continue
        bar()
        results.append({
          "language": element['lang'],
          "seed": seed,
          "problem statement": problem_statement,
          "solution": solution,
          "model": "Meta-Llama-3-70B-Instruct"
        })
        if(i % 100 == 0):
            with open('llama-outputs-3.json', 'w') as fp:
              json.dump(results, fp)
        if(i == 59400):
            break
      i=i+1
with open('llama-outputs-3.json', 'w') as fp:
    json.dump(results, fp)
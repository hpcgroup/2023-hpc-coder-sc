""" Use Gemini API to create a synthetic dataset of performance data.
    The model be given an inspiration prompt and be asked to write a 
    problem statement and one solution.
    Prompts are divided into 4 sections: 
    original(where we just get a problem and solution based on the code snippet), 
    optimization(where we ask the model to provide an optimization problem and solution),
    translation(translating from one language to another eg. CUDA to openmp),
    parallelization(parallelize serial code)
"""

# std imports
from collections import Counter
import json
import os
import time
from typing import Optional

# tpl imports
from alive_progress import alive_bar
import datasets
import google.generativeai as genai

seed_dataset_id = 'hpcgroup/hpc-stack-seeds'

max_new_tokens = 2048
random_seed = 42
language_counter = Counter()
total_samples = 14000
prompt_template_1 = """You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions.
Please gain inspiration from the random code snippet below to create a high-quality programming problem. Be creative. Present your output in two distinct sections: [Problem Description] and [Solution].
Provide code in the [Solution] section that solve the problem you describe in the [Problem Description] section. 
You must use the following code snippet as inspiration for the problem you describe in the [Problem Description] section:
{seed}
Lines for each section of output:
1. [Problem Description]: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.
2. [Solution]: Offer a comprehensive, **correct**, **optimal** solution that accurately addresses the [Problem Description] you provided. The provided code should be **fast** and **efficient**.
"""

prompt_template_2 = """You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions.
Please gain inspiration from the following random code snippet to create a high-quality code optimization problem. Be creative.Present your output in two distinct sections: [Problem Description] and [Solution].
Code snippet for inspiration:
{seed}
lines for each section:
1. [Problem Description]: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. The problem should require optimizing a piece of code. Provide this code to optimize in the problem description. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.
2. [Solution]: Offer a comprehensive, **correct** solution that accurately addresses the [Problem Description] you provided and optimizes the code. Include the optimized code.
"""

prompt_template_3 = """You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions.
Please gain inspiration from the following random code snippet to create a high-quality code translation problem. Present your output in two distinct sections: [Problem Description] and [Solution].
Code snippet for inspiration:
{seed}
lines for each section:
1. [Problem Description]: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. The problem should require translating code between execution models (i.e. translating cuda to openmp or openmp to mpi). Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.
2. [Solution]: Offer a comprehensive, **correct** solution that accurately addresses the [Problem Description] you provided and translates the code. Include the translated code.
"""

prompt_template_4 = """You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions.
Please gain inspiration from the following random code snippet to create a high-quality code parallelization problem. Present your output in two distinct sections: [Problem Description] and [Solution].
Code snippet for inspiration:
{seed}
lines for each section:
1. [Problem Description]: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. The problem should require parallelizing a piece of code. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.
2. [Solution]: Offer a comprehensive, **correct** solution that accurately addresses the [Problem Description] you provided and parallelizes the code. Include the parallel code.
"""

genai.configure(api_key='AIzaSyDoeDru21t5NPzLgiM5IMTsJ1FgOVz6MSw')
config = genai.types.GenerationConfig(
        candidate_count=1,
        temperature=0.6,
        max_output_tokens=max_new_tokens,
)
model = genai.GenerativeModel("gemini-1.0-pro", generation_config=config)

def get_gemini_model_output(model, prompt: str) -> Optional[str]:
    """ Query the Gemini API to get the model output for the given prompt.
    """
    completion = model.generate_content(prompt)
    if completion.candidates[0].finish_reason == 1:
        return completion.text.strip()
    else:
        return None

def postprocess(output: str) -> str:
    """ Postprocess the model output to return the text from each section.
        This is accomplished by finding lines that contain each section header.
    """
    lines = output.splitlines()
    problem_statement_line_idx = [idx for idx, line in enumerate(lines) if '[Problem Description]' in line][0]
    solution_line_idx = [idx for idx, line in enumerate(lines) if '[Solution]' in line][0]

    problem_statement_str = '\n'.join(lines[problem_statement_line_idx+1:solution_line_idx])
    solution_str = '\n'.join(lines[solution_line_idx+1:])
    return problem_statement_str, solution_str

seed_dataset = datasets.load_dataset(seed_dataset_id, split='train', streaming=True).shuffle(seed=random_seed, buffer_size=50)
i = 1
results = []
with alive_bar(total_samples) as bar:
    bar(len(results), skipped=True)
    for element in seed_dataset:
      if(i > 26000):
        seed = element['text']

        try:
            if(i % 4 == 0):
                prompt = prompt_template_4.format(seed=seed)
                output = get_gemini_model_output(model, prompt)
                problem_statement, solution = postprocess(output)
            elif(i % 3 == 0):
                prompt = prompt_template_3.format(seed=seed)
                output = get_gemini_model_output(model, prompt)
                problem_statement, solution = postprocess(output)
            elif(i % 2 == 0):
                prompt = prompt_template_2.format(seed=seed)
                output = get_gemini_model_output(model, prompt)
                problem_statement, solution = postprocess(output)
            else:
                prompt = prompt_template_1.format(seed=seed)
                output = get_gemini_model_output(model, prompt)
                problem_statement, solution = postprocess(output)
        except Exception as e:
            print("Error:{e}")
            print('Sleeping for 5 seconds...')
            time.sleep(5)
            continue
        results.append({
            "language": element['lang'],
            "seed": seed,
            "problem statement": problem_statement,
            "solution": solution,
            "model": "gemini-1.0-pro"
        })
        bar()
        if i % 100 == 0:
            print(i)
            time.sleep(5)

            # cache it intermittently in case something fails, so we don't lose
            # expensive API calls
            with open('gemini-outputs-2.json', 'w') as fp:
                json.dump(results, fp)
      i=i+1

datasets.Dataset.from_list(results).push_to_hub('hpcgroup/hpc-synthetic-gemini',token='HF write token')

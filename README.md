# HPC-Coder-v2

The HPC-Coder-v2-6.7b model is an HPC code LLM fine-tuned on an instruction
dataset catered to common HPC topics such as parallelism, optimization,
accelerator porting, etc. This version is a fine-tuning of the [Deepseek Coder
6.7b](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base) model. It is
fine-tuned on the
[hpc-instruct](https://huggingface.co/datasets/hpcgroup/hpc-instruct),
[oss-instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K),
and
[evol-instruct](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1)
datasets. We utilized the distributed training library
[AxoNN](https://github.com/axonn-ai/axonn) to fine-tune in parallel across many
GPUs.

HPC-Coder-v2-6.7b is the best performing LLM under 30b parameters on the
[ParEval](https://github.com/parallelcodefoundry/ParEval) parallel code
generation benchmark in terms of _correctness_ and _performance_. It scores
similarly to 34B and commercial models like Phind-V2 and GPT-4 on parallel code
generation.

## Using HPC-Coder-v2

The model is provided as a standard huggingface model with safetensor weights.
The weights are available on
[huggingface](https://huggingface.co/hpcgroup/hpc-coder-v2-6.7b). It can be used
with [transformers
pipelines](https://huggingface.co/docs/transformers/en/main_classes/pipelines),
[vllm](https://github.com/vllm-project/vllm), or any other standard model
inference framework. HPC-Coder-v2 is an instruct model and prompts need to be
formatted as instructions for best results. It was trained with the following
instruct template:

```md
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:

```

## Quantized Models

4 and 8 bit quantized weights are available in the GGUF format for use with
[llama.cpp](https://github.com/ggerganov/llama.cpp). The 4 bit model requires
~3.8 GB memory and can be found
[here](https://huggingface.co/hpcgroup/hpc-coder-v2-6.7b-Q4_K_S-GGUF). The 8 bit
model requires ~7.1 GB memory and can be found
[here](https://huggingface.co/hpcgroup/hpc-coder-v2-6.7b-Q8_0-GGUF). Further
information on how to use them with llama.cpp can be found in [its
documentation](https://github.com/ggerganov/llama.cpp).

## Evaluation

We evaluated the model on the ParEval benchmark for parallel code generation. It
scores a pass@1 of 31.17 on parallel code generation tasks including OpenMP,
MPI, MPI+OpenMP, CUDA, HIP, and Kokkos. This is the best performing open-source
model on ParEval under 30B parameters. Furthermore, it performs similarly to the
34B parameter model Phind-V2-34B (pass@1 = 32.12) and GPT-4 (pass@1 = 37.75).
Check out [ParEval](https://github.com/parallelcodefoundry/ParEval) for more
information.

from argparse import ArgumentParser

def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_id", default="deepseek-ai/deepseek-coder-6.7b-base", type=str,
                       help="name of huggingface transformers model you want to run")
    parser.add_argument("--seed", type=int, default=123456, 
                        help="random seed")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"],
                        help="data type for running inference", default="fp16")
    parser.add_argument("--use-flash-attention", action='store_true',
                        help="Use Flash Attention for faster training")
    parser.add_argument("--global-batch-size", type=int, default=16, 
                        help="Global Batch Size")
    parser.add_argument("--gradient-acc-steps", type=int, default=1, 
                        help="Gradient Accumulation Steps")
    parser.add_argument("--sequence-length", type=int, default=256, 
                        help="Sequence Length")
    parser.add_argument("--disable-axonn", action='store_false', dest='use_axonn',
                        help="Disable AxoNN's Tensor Paralellism")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Interval for logging train loss")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save model weights after every --save-every iterations")
    parser.add_argument("--check-max-mem-usage", action='store_true',
                        help="Pad all sequences to the --sequence-length to get the maximum memory usage.")
    return parser

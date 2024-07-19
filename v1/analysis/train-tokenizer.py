''' This script is taken from 
    https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/bpe_training.py
    with some slight modifications. It will train a new tokenizer on our dataset.
'''
from argparse import ArgumentParser
from os import environ

from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer, HfArgumentParser
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


parser = ArgumentParser(description='script to train new tokenizer')
parser.add_argument('--n-examples', type=int, default=10000, help='number of examples to train on')
parser.add_argument('--text-column', type=str, default='text', help='text feature name')
parser.add_argument('--base-tokenizer', type=str, default='gpt2', help='name of base tokenizer')
parser.add_argument('--dataset', type=str, required=True, help='what dataset to train on')
parser.add_argument('--vocab-size', type=int, default=1024, help='number of tokens in vocab')
parser.add_argument('--tokenizer-name', type=str, default='hpc-tok', help='output tokenizer name')
args = parser.parse_args()

#environ['TOKENIZERS_PARALLELISM'] = '0'
environ['OMP_NUM_THREADS'] = '8'


# Iterator for Training
def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, args.n_examples, batch_size)):
        yield [next(iter_dataset)[args.text_column] for _ in range(batch_size)]

# Base tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
base_vocab = list(bytes_to_unicode().values())
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
#dataset = load_dataset('json', data_files=args.dataset, split="train", streaming=True)
dataset = load_dataset(args.dataset, split='train', streaming=True)
iter_dataset = iter(dataset)


# Training and saving
new_tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(), vocab_size=args.vocab_size, initial_alphabet=base_vocab
)
new_tokenizer.save_pretrained(args.tokenizer_name)#, push_to_hub=args.push_to_hub)
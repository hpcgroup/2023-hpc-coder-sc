''' Generate a bunch of results for codegeneration.
    author: Daniel Nichols
    date: November 2022
'''
# std imports
from argparse import ArgumentParser
from itertools import product
import json
from typing import Iterable, Optional

# tpl imports
from alive_progress import alive_it
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, pipeline, StoppingCriteria, set_seed


class PromptDataset(Dataset):
    ''' PyTorch dataset that simply wraps a list of strings. They do not have to have the same length.
    '''

    def __init__(self, prompts):
        super().__init__()
        self.prompts_ = prompts
    
    def __len__(self):
        return len(self.prompts_)
    
    def __getitem__(self, idx): 
        return self.prompts_[idx]


def has_balanced_brackets(text : str, left_bracket : str = '{', right_bracket : str = '}') -> bool:
    ''' Check if string has balanced brackets.
        taken from: https://stackoverflow.com/a/38834249/3769237

        Arguments:
            text: string to check for balanced brackets in.
            left_bracket: left bracket to balance
            right_bracket: right bracket to balance

        Returns:
            true if left_bracket and right_bracket are balanced
    '''
    stack = []
    balanced = True
    index = 0
    while index < len(text) and balanced:
        token = text[index]
        if token == left_bracket:
            stack.append(token)
        elif token == right_bracket:
            if len(stack) == 0:
                balanced = False
            else:
                stack.pop()

        index += 1

    return balanced and len(stack) == 0


class BalancedBracketsCriteria(StoppingCriteria):
    ''' extension of transformers' text-generation stopping criteria.
        Stops either when function is complete (i.e. { and } are balanced) or when max_length is surpassed, whichever
        happens first. 

        _Note:_ This is a slow stopping criteria, but it's much faster than continually running model inference when 
        it does not need to be run anymore.
    '''

    def __init__(self, max_length : int, tokenizer, left_bracket : str = '{', right_bracket : str = '}'):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.left_bracket = left_bracket
        self.right_bracket = right_bracket
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] > self.max_length:
            # already to long, early stop
            return True

        # return true if {} are balanced i.e. the function is complete
        return all(
            has_balanced_brackets(
                self.tokenizer.decode(t), 
                left_bracket=self.left_bracket, 
                right_bracket=self.right_bracket
            ) for t in input_ids)


def get_predictions(
    prompts : Iterable[str],
    generator,
    num_samples : int = 100,
    top_p : float = 0.95,
    top_k : int = 50,
    temperature : float = 0.2,
    min_len : int = 50,
    max_len : int = 500,
    batch_size : int = 1,
    tokenizer = None
) -> Iterable[dict]:
    ''' Get prediction outputs from model.

        Arguments:
            prompts: list of text prompts to generate text for
            generator: transformers pipeline object
            num_samples: how many samples to generate for each prompt
            top_p: probability for nucleus sampling
            top_k: k for top-k sampling
            temperature: inference temperature
            min_len: minimum generation length
            max_len: maximum generation length
            batch_size: [deprecated] how many samples to process at once
            tokenizer: HF tokenizer to be passed to stopping criteria

        Returns:
            a list of result objects that store the results as well as meta-data
    '''

    prompts = [ p for p in prompts for _ in range(num_samples) ]
    ds = PromptDataset(prompts)

    stopping_criteria = BalancedBracketsCriteria(max_len, tokenizer)

    gen_output = generator(
        ds,
        return_full_text=True,
        do_sample=True,
        max_new_tokens=max_len,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=50256, # suppress error
        stopping_criteria=[stopping_criteria]
    )

    generated_text = []
    bar = alive_it(gen_output, total=len(gen_output)*batch_size, title=f'Temperature {temperature}')
    for g in bar:
        texts = list(map(lambda x: x['generated_text'], g))
        generated_text.extend( texts )
    
    results = []
    for prompt, text in zip(prompts, generated_text):
        result = {
            'prompt': prompt,
            'generated_text': text,
            'num_samples': num_samples,
            'min_len': min_len,
            'max_len': max_len,
            'top_p': top_p,
            'top_k': top_k,
            'temperature': temperature,
        }
        results.append( result )

    return results


def main():
    parser = ArgumentParser(description='Generate code samples for a set of test problems.')
    parser.add_argument('-m', '--model', type=str, required=True, help='path to model or HF hub model name')
    parser.add_argument('-t', '--tokenizer', type=str, required=True, help='path to tokenizer of HF hub name')
    parser.add_argument('-i', '--input', type=str, required=True, help='json file with all the test prompts')
    parser.add_argument('-o', '--output', type=str, required=True, default='results.json', help='output path')
    parser.add_argument('--cache-dir', type=str, default='~/.cache/huggingface', help='path to HF cache')
    parser.add_argument('-k', '--num-samples', type=int, default=100, help='how many samples to generate')
    parser.add_argument('--min-len', type=int, default=50, help='Minimum length to generate.')
    parser.add_argument('--max-len', type=int, default=350, help='Maximum length to generate.')
    parser.add_argument('--top-k', type=int, default=50, help='Number of samples to use in top-k sampling.')
    parser.add_argument('--top-p', type=float, default=0.95, help='Fraction to use in nucleas sampling.')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0], 
        help='Sampling temperatures to try.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size to feed to inference.')
    parser.add_argument('--device', type=int, default=-1, help='Where to run model')
    parser.add_argument('--max-sequence-length', type=int, default=1024, help='maximum sequence length of model')
    args = parser.parse_args()

    set_seed(42)

    # read input prompts
    with open(args.input, 'r') as fp:
        prompts = json.load(fp)
    
    print(f'Running inference on {len(prompts)} total prompts.')

    # get hf models
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    generator = pipeline('text-generation', model=args.model, tokenizer=tokenizer, framework='pt', device=args.device,
        batch_size=args.batch_size)

    # filter prompts
    filtered_prompts = []
    bar = alive_it(prompts, title='Removing Long Prompts')
    for prompt in bar:
        tokens = tokenizer(prompt['prompt'])['input_ids']
        if len(tokens) > args.max_sequence_length:
            print('Skipping prompt \'{}\' as its number of tokens exceeds max_sequence_length ({} > {}).'.format(
                prompt['name'],
                len(tokens),
                args.max_sequence_length
            ))
        else:
            filtered_prompts.append( prompt )
    
    prompts = filtered_prompts

    # run tests for all temperatures
    for temperature in args.temperatures:
        
        prompt_results = get_predictions(
            [p['prompt'] for p in prompts],
            generator,
            num_samples=args.num_samples,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=temperature,
            min_len=args.min_len,
            max_len=args.max_len,
            batch_size=args.batch_size,
            tokenizer=tokenizer
        )

        with open(args.output, 'a') as fp:
            for r in prompt_results:
                # extra meta-data
                r['model'] = args.model
                r['tokenizer'] = args.tokenizer
                r['name'] = prompt['name']

                # write out
                json.dump(r, fp, ensure_ascii=True)
                fp.write('\n')


if __name__ == '__main__':
    main()

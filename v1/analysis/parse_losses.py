''' Parse the output of the training script for losses and output csv.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from os import PathLike
from typing import Optional, Iterable
from collections import deque
from csv import QUOTE_NONNUMERIC
import json
import math
from os.path import join as path_join

# tpl imports
import pandas as pd


def parse_output(
    results_txt_files: Iterable[PathLike], 
    add_perplexity: bool = True,
    samples_per_step: int = 2,
    model_name : Optional[str] = None
) -> pd.DataFrame:
    ''' Parse the output of a training run.

        Args:
            results_txt_file: paths to the text outputs of training runs
            add_perplexity: calculate perplexity for result if it's not already there
            samples_per_step: how many samples per step were computed
            model_name: include the name of the model in the dataframe
        
        Returns:
            Two dataframes t,v  -- the first is the training results and the second eval results
    '''
    LINE_START_KEY = '{\'loss\':'
    EVAL_START_KEY = '{\'eval_loss\':'
    BAD_ESC_STR = ''.join(chr(o) for o in [27, 91, 65])

    results, eval_results = [], []
    for text_file in results_txt_files:
        with open(text_file, 'r', encoding='ascii', errors='ignore') as fp:
            prev_lines = deque(3*[None], maxlen=3)
            last_steps = 0
            for line in fp:                
                line = line.strip().replace(BAD_ESC_STR, '')
                if line.startswith(LINE_START_KEY) or LINE_START_KEY in line:
                    obj = json.loads(line.replace('\'', '"'))
                    try:
                        steps = int( prev_lines[-1].split()[2].split('/')[0] )
                    except:
                        steps = int( prev_lines[-1].split()[1].split('/')[0] )
                    last_steps = steps

                    if add_perplexity and 'perplexity' not in obj and 'loss' in obj:
                        obj['perplexity'] = math.exp(obj['loss'])
                    
                    if 'steps' not in obj:
                        obj['steps'] = steps

                    if 'samples' not in obj:
                        obj['samples'] = obj['steps'] * samples_per_step

                    if model_name is not None:
                        obj['model'] = model_name
                    
                    results.append( obj )
                
                elif line.startswith(EVAL_START_KEY) or EVAL_START_KEY in line:
                    obj = json.loads(line.replace('\'', '"'))
                    #steps = int( prev_lines[-1].split()[1].split('/')[0] )
                    steps = last_steps

                    if add_perplexity and 'perplexity' not in obj and 'eval_loss' in obj:
                        obj['perplexity'] = math.exp(obj['eval_loss'])

                    if 'steps' not in obj:
                        obj['steps'] = steps

                    if 'samples' not in obj:
                        obj['samples'] = obj['steps'] * samples_per_step

                    if model_name is not None:
                        obj['model'] = model_name

                    eval_results.append( obj )

                
                if line.strip() != '':
                    prev_lines.append(line)
    
    results.sort(key=lambda x: x['samples'])
    eval_results.sort(key=lambda x: x['samples'])

    return pd.DataFrame( results ), pd.DataFrame( eval_results )


def main():
    parser = ArgumentParser(description='Scrape loss and accuracy from training results.')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='training output files')
    parser.add_argument('-o', '--output', type=str, required=True, help='where to write output csv')
    parser.add_argument('--eval-output', type=str, required=True, help='where to write eval output csv')
    parser.add_argument('--samples-per-step', type=int, default=8, help='how many samples are computed per step')
    parser.add_argument('--model-name', type=str, help='Name of model being trained to include in data.')
    args = parser.parse_args()

    results, eval_results = parse_output(args.input, samples_per_step=args.samples_per_step, model_name=args.model_name)
    results.to_csv(args.output, index=False, quoting=QUOTE_NONNUMERIC)

    eval_results.columns = eval_results.columns.str.removeprefix('eval_')
    eval_results.to_csv(args.eval_output, index=False, quoting=QUOTE_NONNUMERIC)


if __name__ == '__main__':
    main()

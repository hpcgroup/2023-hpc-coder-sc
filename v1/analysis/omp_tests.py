''' Run the OpenMP auto-complete tests.
    author: Daniel Nichols
    date: November 2022
'''
# std imports
from argparse import ArgumentParser
from os import environ
from typing import Tuple

# tpl imports
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, set_seed


def get_loop_text(text : str, end_loop_token : str = '<LOOP-END> ') -> str:
    ''' 
    '''
    return (text.split(end_loop_token)[0] + end_loop_token).strip()


def get_predicted_omp(text : str, end_loop_token : str = '<LOOP-END> ', end_pragma_token = '<OMP-END>') -> str:
    '''
    '''
    pragma = text.split(end_pragma_token)[0]
    return pragma


def chunks(lst, n):
    '''Yield successive n-sized chunks from lst.'''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def is_correct_pragma(
    generated_text : str, 
    real_pragma : str
) -> bool:
    '''
    '''
    if '<OMP-END>' not in generated_text and generated_text.startswith('#pragma omp parallel for'):
        generated_text = generated_text.split('\n')[0].strip()
    elif '<LOOP-END> #pragma omp' in generated_text:
        generated_text = generated_text.split('<LOOP-END>')[1].strip()
    else:
        generated_text = generated_text.split('<OMP-END>')[0].strip()

    print(f'Predicted: \'{generated_text}\'')

    return generated_text == real_pragma


def test(
    generator, 
    data, 
    true_results,
    max_len : int = 200,
    top_k : int = 50,
    top_p : float = 0.95,
    num_samples : int = 10,
    temperature : float = 0.2
) -> float:
    '''
    '''
    results = []
    try:
        for idx, d in enumerate( data ):
            #print(f'{idx}: \'{d}\'', flush=True)
            tmp_results = generator(
                d,
                return_full_text=False,
                do_sample=True, 
                max_new_tokens=max_len,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_samples,
                temperature=temperature
            )
            results.append( tmp_results )
    except Exception as err:
        print(f'Error during inference: \'{err}\'', flush=True)
        print(results)
        print(d, flush=True)
        return 0, 0

    num_correct, num_incorrect = 0, 0
    for idx, (result, true_result) in enumerate( zip(results, true_results) ):
        print(f'Sample {idx}:')
        print(f'Real: \'{true_result}\'')

        is_correct = any( is_correct_pragma(s['generated_text'], true_result.strip()) for s in result )

        if is_correct:
            print('CORRECT')
            num_correct += 1
        else:
            print('INCORRECT')
            num_incorrect += 1
        
        print()
    
    return num_correct / (num_correct + num_incorrect), (num_correct + num_incorrect) 


def main():
    parser = ArgumentParser(description='Test a models OpenMP pragma prediction.')
    parser.add_argument('-m', '--model', type=str, required=True, help='path to model or HF hub model name')
    parser.add_argument('--tokenizer', type=str, required=True, help='text tokenizer')
    parser.add_argument('--cache-dir', type=str, default='~/.cache/huggingface', help='path to HF cache')
    parser.add_argument('-k', '--num-samples', type=int, default=1, help='how many samples to generate')
    parser.add_argument('--min-len', type=int, default=50, help='Minimum length to generate.')
    parser.add_argument('--max-len', type=int, default=150, help='Maximum length to generate.')
    parser.add_argument('--top-k', type=int, default=50, help='Number of samples to use in top-k sampling.')
    parser.add_argument('--top-p', type=float, default=0.95, help='Fraction to use in nucleas sampling.')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size to feed to inference.')
    parser.add_argument('--device', type=int, default=-1, help='Where to run model')
    args = parser.parse_args()

    # environment setup
    environ['TOKENIZERS_PARALLELISM'] = '0'
    environ['OMP_NUM_THREADS'] = '64'
    set_seed(42)

    val_dataset = load_dataset(
        'hpcgroup/omp-for-loops',
        split='train[:5%]',
        cache_dir=args.cache_dir
    )

    # create pipeline
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    generator = pipeline('text-generation', model=args.model, tokenizer=tokenizer, framework='pt', device=args.device)

    inference_batches, true_outputs = [], []
    for sample in val_dataset:
        omp_pragma = sample['omp_pragma_line']
        loop = get_loop_text(sample['text'])

        toks = tokenizer(loop)['input_ids']
        if len(toks) >= 1024:
            continue

        inference_batches.append( loop )
        true_outputs.append( omp_pragma )
        
    accuracy, total = test(generator, inference_batches, true_outputs, max_len=args.max_len, top_k=args.top_k, 
        top_p=args.top_p, num_samples=args.num_samples, temperature=args.temperature)
    
    print('Accuracy: {}% ({} tested)'.format(accuracy * 100.0, total))


if __name__ == '__main__':
    main()

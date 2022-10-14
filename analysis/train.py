''' Train LLM on source code data.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from typing import Iterable
from os import PathLike

# tpl imports
from datasets import load_dataset

# local imports
from load_dataset import get_source_filenames, get_source_file_size, get_loc, filter_bad_encoding, filter_duplicates


def get_args():
    ''' Parse the command line arguments and return the object with them as properties.
    '''
    parser = ArgumentParser(description='Train a LLM on source code data')
    parser.add_argument('--input', type=str, required=True, help='root of textual source data or path to pkl of ' +
        'filenames list')
    parser.add_argument('--cache-fnames', type=str, help='cache the filenames to this path')
    parser.add_argument('--deduplicate', action='store_true', help='If provided, then data will be deduplicated')
    parser.add_argument('--model', type=str, choices=['NeoX', 'GPT2'], help='What model to train')
    return parser.parse_args()


def print_source_file_stats(fnames: Iterable[PathLike]):
    ''' Print meta-data about source files such as # files, LOC, and memory size.
    
        Args:
            fnames: File names to compute statistics over
    '''
    loc = get_loc(fnames)
    size = get_source_file_size(fnames)

    print('# source files: {:,}'.format(len(fnames)))
    print('LOC: {:,}'.format(loc))
    print('Dataset size: {:.3g} GB'.format(size / (1<<30)))


def get_dataset(dataset_path: PathLike, deduplicate: bool = True, fnames_cache_output: Optional[PathLike] = None,
    print_stats: bool = True) -> DatasetDict:
    ''' Fetch the dataset from dataset_path and return a huggingface DatasetDict object.

        Args:
            dataset_path:
            deduplicate:
            fnames_cache_output: fnames
            print_stats: If true, then print summary statistics of data set.
    '''
    if isdir(dataset_path):
        # read filenames from root
        fnames = get_source_filenames(dataset_path)
        fnames = filter_bad_encoding(fnames)

        if fnames_cache_output:
            with open(fnames_cache_output, 'wb') as fp:
                pickle.dump(fnames, fp)
    else:
        # read filenames from pickle
        with open(dataset_path, 'rb') as fp:
            fnames = pickle.load(fp)

    if deduplicate:
        fnames = filter_duplicates(fnames)
    
    if print_stats:
        print_source_file_stats(fnames)
        
    return load_dataset('text', name='HPC-Source-Dataset', data_files=fnames, encoding='utf-8', sample_by='document')
    
def main():
    args = get_args()

    dataset = get_dataset(args.input, deduplicate=args.deduplicate, fnames_cache_output=args.cache_fnames)

    dataset = load_dataset("text", name='HPC Source Dataset', data_files=fnames, encoding='utf-8')
    print(dataset)



if __name__ == '__main__':
    main()


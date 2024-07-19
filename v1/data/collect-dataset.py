''' Create a jsonl text dataset for the list of source files.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from __future__ import annotations
from argparse import ArgumentParser
import json

# local imports
from dataset_utils import get_source_filenames, filter_bad_encoding, filter_by_size, filter_duplicates, \
    print_source_file_stats


def get_args():
    parser = ArgumentParser(description='Create compact dataset representation')
    parser.add_argument('--root', type=str, required=True, help='root to start searching for source files')
    parser.add_argument('-o', '--output', type=str, required=True, help='output path')
    return parser.parse_args()


def main():
    args = get_args()

    # retrieve dataset
    fnames = get_source_filenames(args.root)
    fnames = filter_bad_encoding(fnames)
    fnames = filter_by_size(fnames, min_tokens=15)
    fnames = filter_duplicates(fnames)
    print_source_file_stats(fnames)

    # write out json lines file
    with open(args.output, 'w') as fp:
        for fname in fnames:
            result = {'filename': fname}
            with open(fname, 'r') as tmp_fp:
                result['text'] = tmp_fp.read()
            
            json.dump(result, fp, ensure_ascii=False)
            fp.write('\n')


if __name__ == '__main__':
    main()
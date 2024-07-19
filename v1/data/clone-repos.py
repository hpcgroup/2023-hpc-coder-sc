''' Clone a list of repos into a specified location.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from os.path import join as path_join
from pathlib import Path
from typing import Optional
from os import PathLike

# tpl imports
import pandas as pd
from alive_progress import alive_it
from git import Repo


def clone(url: str, root: PathLike, dirname: Optional[PathLike] = None):
    ''' Clone the specified git url.
    
        Args:
            url: github url to clone
            root: where to store cloned repos
            dirname: subdirectory of root to clone into
    '''
    if dirname is None:
        raise NotImplementedError('\'clone\' expects a dirname.')

    dest_path = Path(path_join(root, dirname))
    if dest_path.is_dir():
        return
    
    # make the directory
    dest_path.mkdir(parents=True, exist_ok=False)
    
    # clone
    Repo.clone_from(url, dest_path)


def main():
    parser = ArgumentParser(description='Clone the repositories in a dataframe into a directory root.')
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Where to find dataset of repos')
    parser.add_argument('--root', type=str, required=True, help='Root directory to clone into')
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    pairs = list(zip(df['full_name'], df['clone_url']))
    bar = alive_it(pairs, title='Cloning Repos')
    for full_name, clone_url in bar:
        clone(clone_url, args.root, dirname=full_name)


if __name__ == '__main__':
    main()


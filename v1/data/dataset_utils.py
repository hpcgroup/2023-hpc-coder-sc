''' Tools for loading the dataset
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from __future__ import annotations
from typing import Iterable
from os import PathLike
from os.path import splitext
import hashlib

# tpl imports
from alive_progress import alive_it


# C/C++ related extensions to include in dataset
C_CPP_EXTENSIONS = ['C', 'cc', 'cxx', 'cpp', 'c', 'h', 'hh', 'hpp', 'H', 'hxx', 'Hxx', 'HXX']


def get_source_filenames(root: PathLike, extensions: Iterable[str] = C_CPP_EXTENSIONS, show_progress: bool = True
) -> list[PathLike]:
    ''' return a list of all the filenames of source files with the given extensions in root.

        Args:
            root: where to start searching for files. Is searched recursively.
            extensions: what extensions define the source files. C/C++ extensions by default.
            show_progress: If true, then display a progress bar.

        Returns:
            A list of paths to all the source files.
    '''
    from os.path import join as path_join, isdir, exists
    from os import walk

    get_extension = lambda x: splitext(x)[-1][1:]

    def is_valid_source_file(fname: PathLike) -> bool:
        return (get_extension(fname) in extensions) and (not isdir(fname)) and (exists(fname)) and \
            (all(c not in fname for c in ['[', ']']))

    # I've found os.walk to be ~2-3x faster at this task than glob.glob
    all_files = []
    vals = alive_it(walk(root), title='Searching for source files'.rjust(26)) if show_progress else walk(root)
    for rt, _, files in vals:
        all_files.extend( [path_join(rt, f) for f in files if is_valid_source_file(path_join(rt, f))] )

    return all_files


def filter_bad_encoding(fnames: Iterable[PathLike], show_progress: bool = True) -> list[PathLike]:
    ''' Remove files with non utf-8 encodings.

        Args:
            fnames: a list of filenames to filter.
            show_progress: If true, then display a progress bar.

        Returns:
            A copy of fnames with files that contained non-utf-8 characters filtered out.
    '''
    results = []
    vals = alive_it(fnames, title='Removing non-utf-8'.rjust(26)) if show_progress else fnames
    for f in vals:
        try:
            for _ in open(f, 'r', encoding='utf-8'):
                pass
            results.append(f)
        except UnicodeDecodeError:
            pass
    return results


def filter_by_size(fnames: Iterable[PathLike], min_mb: int = 0, max_mb: int = 1, min_tokens: int = 50, 
    show_progress: bool = True
) -> list[PathLike]:
    ''' Remove files based on size of file and number of tokens.
        Args:
            fnames: List of filenames to filter
            min_mb: minimum number of MB to allow
            max_mb: maximum number of MB to allow
            min_tokens: exclude files with less tokens (split by whitespace)
    '''
    from os.path import getsize
    result = []
    vals = alive_it(fnames, title='Filtering by size'.rjust(26)) if show_progress else fnames
    
    for fname in vals:
        mb = getsize(fname) / (1024 ** 2)
        if mb < min_mb or mb > max_mb:
            continue
        
        num_tokens = 0
        with open(fname, 'r') as fp:
            for line in fp:
                num_tokens += len( line.split() )
                if num_tokens >= min_tokens:
                    break
        
        if num_tokens < min_tokens:
            continue

        result.append( fname )
    
    return result


def _file_hash(fname: PathLike) -> str:
    ''' Compute hash of contents of fname. Method body from https://stackoverflow.com/a/44873382/3769237.

        Args:
            fname: path to file
        
        Returns:
            sha256 hash of binary contents of fname
    '''
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(fname, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def filter_duplicates(fnames: Iterable[PathLike], show_progress: bool = True) -> list[PathLike]:
    ''' Perform deduplication.

        Args:
            fnames: names of files to deduplicate
            show_progress: If True, then display a progress bar.
        
        Returns:
            fnames with the duplicates filtered out
    '''

    hashes = set()
    unique_fnames = []
    bar = alive_it(fnames, title='Deduplicating'.rjust(26)) if show_progress else fnames
    for fname in bar:
        fhash = _file_hash(fname)
        if fhash not in hashes:
            hashes.add( fhash )
            unique_fnames.append( fname )

    return unique_fnames


def get_loc(flist: Iterable[PathLike], show_progress: bool = True) -> int:
    ''' Returns the total number of lines in all the files in flist.

        Args:
            flist: a list of filenames to count LOC in.
            show_progress: If true, then display a progress bar.
        
        Returns:
            The total LOC summed over all the files.
    '''
    #import subprocess
    LOC = 0
    vals = alive_it(flist, title='Counting LOC'.rjust(26)) if show_progress else flist
    for fname in vals:
        #LOC += int(subprocess.check_output(['wc', '-l', fname]).split()[0])
        LOC += sum(1 for _ in open(fname, 'r', errors='ignore'))
    return LOC


def get_loc_per_extension(flist: Iterable[PathLike], show_progress: bool = True) -> int:
    ''' Returns the total number of lines in all the files in flist per extension.

        Args:
            flist: a list of filenames to count LOC in.
            show_progress: If true, then display a progress bar.
        
        Returns:
            The total LOC summed over all the files stored in buckets in a dict.
    '''
    get_extension = lambda x: splitext(x)[-1]

    LOC = {}
    vals = alive_it(flist, title='Counting LOC'.rjust(26)) if show_progress else flist
    for fname in vals:
        ext = get_extension(fname)
        if ext not in LOC:
            LOC[ext] = 0

        LOC[ext] += sum(1 for _ in open(fname, 'r', errors='ignore'))
        
    return LOC


def get_source_file_size(flist: Iterable[PathLike], show_progress: bool = True) -> int:
    ''' Return the data set size based on a list of fnames in bytes.

        Args:
            flist: a list of filenames to sum the size over.
            show_progress: If true, then display a progress bar.

        Returns:
            The total number of bytes that flist files takes up.
    '''
    from os.path import getsize

    num_bytes = 0
    vals = alive_it(flist, title='Calculating dataset size'.rjust(26)) if show_progress else flist
    for fname in vals:
        num_bytes += getsize(fname)
    return num_bytes


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

''' Some summary plots of the repository data.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from os import PathLike
from typing import Tuple, Optional, Union

# tpl imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def extensions_histogram(ds: pd.DataFrame, fname: PathLike):
    ''' Plot a histogram of the extensions over the repos.
    
        Args:
            ds: repo metadata dataset
    '''
    from os.path import splitext
    from dataset_utils import get_source_filenames, filter_bad_encoding, filter_duplicates, filter_by_size

    ROOT = '/afs/shell.umd.edu/project/bhatele-lab/user/dnicho/code-ml/data/repos'
    fnames = get_source_filenames(ROOT)
    fnames = filter_duplicates( filter_by_size( filter_bad_encoding(fnames), min_tokens=15 ) )

    exclude_ext = ['.cxx', '.hh', '.H', '.hxx']

    get_extension = lambda x: splitext(x)[-1]
    extensions = list( filter(lambda x: x not in exclude_ext, map(get_extension, fnames)) )

    plt.clf()
    sns.set()
    hist_fig = sns.histplot(x=extensions)
    hist_fig.set_title('File Type Distribution')
    hist_fig.set_xlabel('File Extension')
    fig = hist_fig.get_figure()
    fig.savefig(fname, bbox_inches='tight')


def loc_histogram(ds: pd.DataFrame, fname: PathLike):
    ''' Plot a histogram of the extensions over the repos.

        Args:
            ds: repo metadata dataset
    '''
    from dataset_utils import get_source_filenames, get_loc_per_extension, filter_bad_encoding, filter_duplicates, \
        filter_by_size
    
    ROOT = '/afs/shell.umd.edu/project/bhatele-lab/user/dnicho/code-ml/data/repos'
    fnames = filter_duplicates( filter_by_size( filter_bad_encoding( get_source_filenames(ROOT) ), min_tokens=15 ) )

    loc = get_loc_per_extension(fnames)

    exclude_ext = ['.cxx', '.hh', '.H', '.hxx']
    for ext in exclude_ext:
        if ext in loc:
            del loc[ext]

    plt.clf()
    sns.set()
    hist_fig = sns.barplot(x=list(loc.keys()), y=list(loc.values()), color='b')
    hist_fig.set_title('Distribution of LOC by File Type')
    hist_fig.set_xlabel('File Extension')
    hist_fig.set_ylabel('LOC')
    fig = hist_fig.get_figure()
    fig.savefig(fname, bbox_inches='tight')


def plot_histogram(data: pd.DataFrame, column: str, fname: PathLike, nbins: Union[int,str] = 'auto', 
    title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None, 
    log_scale: Union[int, bool, Tuple[Union[int,bool], Union[int,bool]]] = False):
    ''' Plot a histogram of 'column' in data. Writes out the histogram to 'fname'.

        Args:
            data: DataFrame to read data from
            column: what column of 'data' to use for histogram
            fname: where to write output file
            nbins: number of histogram bins
            title: title of plot
            xlabel: xlabel of plot
            ylabel: ylabel of plot
            log_scale: how to log scale the axes. Either False, an integer, or tuple of integers (or mix bool/int).
    '''
    plt.clf()
    sns.set()
    hist_fig = sns.histplot(data=data, x=column, bins=nbins, log_scale=log_scale)
    if title:
        hist_fig.set_title(title)
    if xlabel:
        hist_fig.set_xlabel(xlabel)
    if ylabel:
        hist_fig.set_ylabel(ylabel)
    hist_fig.get_figure().savefig(fname)


def tags_wordcloud(ds: pd.DataFrame, fname: PathLike):
    ''' Plot a wordcloud of the tags of the repos.

        Args:
            ds: repo metadata dataset
    '''
    from wordcloud import WordCloud
    def to_list(x):
        return [str(s.strip('\'')) for s in x.strip('][').split(', ') if s != '']

    plt.clf()
    series_as_list = ds['topics'].values.tolist()
    series_as_list = [to_list(x) for x in series_as_list]
    all_tags = [item for sublist in series_as_list for item in sublist]
    
    wc = WordCloud(background_color='white', width=800, height=400).generate(' '.join(all_tags))
    plt.figure(figsize=(20,10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')


def main():
    parser = ArgumentParser(description='script to plot dataset info for repo metadata dataset')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset to use data from')
    parser.add_argument('--languages', type=str, help='create a histogram of languages')
    parser.add_argument('--extensions', type=str, help='create a histogram of language extensions')
    parser.add_argument('--loc', type=str, help='create a histogram of loc per extension')
    parser.add_argument('--sizes', type=str, help='create a histogram of sizes')
    parser.add_argument('--stars', type=str, help='create a histogram of stars')
    parser.add_argument('--watchers', type=str, help='create a histogram of watchers')
    parser.add_argument('--forks', type=str, help='create a histogram of forks')
    parser.add_argument('--tags', type=str, help='create a wordcloud of tags')
    args = parser.parse_args()

    ds = pd.read_csv(args.dataset)

    sns.set_theme()
    if args.languages:
        plot_histogram(ds, 'language', args.languages, title='Repository Main Language Distribution')
    if args.sizes:
        plot_histogram(ds, 'size', args.sizes, title='Repository Size Distribution', xlabel='Repo Size (KB)',
            nbins=15, log_scale=(2,False))
    if args.stars:
        plot_histogram(ds, 'stargazers_count', args.stars, title='Repository Stars Distribution', xlabel='# Stars',
            nbins=15, log_scale=False)
    if args.watchers:
        plot_histogram(ds, 'watchers_count', args.watchers, title='Repository Watchers Distribution', 
            xlabel='# Watchers', nbins=15, log_scale=False)
    if args.forks:
        plot_histogram(ds, 'forks_count', args.forks, title='Repository Forks Distribution', xlabel='# Forks', nbins=15)
    if args.tags:
        tags_wordcloud(ds, args.tags)
    if args.extensions:
        extensions_histogram(ds, args.extensions)
    if args.loc:
        loc_histogram(ds, args.loc)



if __name__ == '__main__':
    main()

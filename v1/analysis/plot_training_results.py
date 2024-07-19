''' Plot training results using data from csv files.
    author: Daniel Nichols
'''
# std imports
from argparse import ArgumentParser
from functools import reduce
from os import PathLike
from typing import Optional
from os.path import join as path_join

# tpl imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot(
    train_data : pd.DataFrame,
    val_data : pd.DataFrame,
    output_path : PathLike,
    xcolumn : str = 'samples', 
    ycolumn : str = 'perplexity',
    seriescolumn : str = 'model',
    xscale : Optional[int] = None,
    title : Optional[str] = None
):
    ''' plot the training loss/perplexity curves

        Args:
            train_data: training dataset
            val_data: validation dataset
            output_path: where to save file
            xcolumn: what column to use for x axis
            ycolumn: what column to use for y axis
            seriescolumn: how to distinguish series on line plot
            xscale: scale value for x-axis
            title: set title of figure if not None
    '''
    assert xcolumn in ['samples', 'steps']
    assert ycolumn in ['loss', 'perplexity', 'accuracy']

    train_data = train_data.copy(deep=True)
    val_data = val_data.copy(deep=True)

    xlabel_prefix = f'{xscale}x ' if xscale else ''
    if xscale:
        train_data[xcolumn] /= xscale
        val_data[xcolumn] /= xscale

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6), sharey=True)

    ax1 = sns.lineplot(data=train_data, x=xcolumn, y=ycolumn, hue=seriescolumn, ax=ax1)
    ax2 = sns.lineplot(data=val_data, x=xcolumn, y=ycolumn, hue=seriescolumn, ax=ax2)

    #ax1.set_ylim((1, None))
    #ax2.set_ylim((1, None))

    for ax, ds in zip([ax1, ax2], [train_data, val_data]):

        series_names = ds[seriescolumn].unique()
        for series_idx, series_name in enumerate(series_names):
            series_ds = ds[ds[seriescolumn] == series_name]

            xpos = (series_ds[xcolumn].values[0], series_ds[xcolumn].values[-1])
            ypos = (series_ds[ycolumn].values[0], series_ds[ycolumn].values[-1])
            color = sns.color_palette()[series_idx]
            for x, y in zip(xpos, ypos): 
                ax.text(x, y, f'{y:.2f}', color=color)

    ax1.set_ylabel(ycolumn.capitalize())
    ax1.set_xlabel(xlabel_prefix + xcolumn.capitalize())
    ax1.get_legend().set_title(seriescolumn.capitalize())

    ax2.set_xlabel(xlabel_prefix + xcolumn.capitalize())
    ax2.get_legend().set_title(seriescolumn.capitalize())

    ax1.set_title('Training')
    ax2.set_title('Validation')

    if title:
        fig.suptitle(title)

    plt.savefig(output_path, bbox_inches='tight')


def main():
    parser = ArgumentParser(description='Plot training results.')
    parser.add_argument('-t', '--training-results', type=str, nargs='+', required=True, help='csv of training results')
    parser.add_argument('-v', '--validation-results', type=str, nargs='+', required=True, 
        help='csv of validation results')
    parser.add_argument('--output-root', type=str, default='figs', help='root of figs directory')
    args = parser.parse_args()

    train_df = pd.concat([pd.read_csv(fpath) for fpath in args.training_results], ignore_index=True)
    val_df = pd.concat([pd.read_csv(fpath) for fpath in args.validation_results], ignore_index=True)

    sns.set(font_scale=1.5, font='DejaVu Sans')
    plot(train_df, val_df, path_join(args.output_root, 'perplexity.png'), ycolumn='perplexity', 
        title='Perplexity During Training', xscale=1000)

    plot(train_df, val_df, path_join(args.output_root, 'loss.png'), ycolumn='loss', title='Loss During Training', 
        xscale=1000)
    

if __name__ == '__main__':
    main()

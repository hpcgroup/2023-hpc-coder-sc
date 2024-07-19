''' Make manual adjustments to repo data set.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from csv import QUOTE_NONNUMERIC
from getpass import getpass
import requests
from typing import Iterable

# tpl imports
import pandas as pd


def get_repo_info(org: str, name: str, access_key: str) -> dict:
    ''' Query GitHub API for metadata about a repo.

        Args:
            org: organization name
            name: repo name within organization
            access_key: GitHub API personal access key

        Returns:
            A dict containing the metadata from the corresponding API request.
    '''
    BASE_URL = 'https://api.github.com/repos'
    DESIRED_ENTRIES = ['name', 'full_name', 'clone_url', 'html_url', 'created_at',
            'updated_at', 'language', 'size', 'stargazers_count', 'watchers_count',
            'forks_count', 'topics', 'visibility', 'forks', 'open_issues', 'watchers']
    query_url = '{}/{}/{}'.format(BASE_URL, org, name)

    try:
        header = {'Authorization': 'token {}'.format(access_key)}
        response = requests.get(query_url, headers=header).json()
    except:
        print('Error fetching GitHub info!')
        return None

    result = { k: response[k] for k in DESIRED_ENTRIES }    
    return result


def append(ds: pd.DataFrame, repos: Iterable[str]) -> pd.DataFrame:
    ''' Fetches metadata for the requested repos and appends them to the data set.
    
        Args:
            ds: dataset to append to
            repos: repositories to get information about

        Returns:
            A new DataFrame with the appended repo information
    '''
    api_token = getpass('GitHub API Token: ')
    
    new_data = []
    for repo in repos:
        parts = repo.split('/')
        if '/' not in repo or len(parts) != 2:
            print('Invalid repo description \'{}\'. Expected <org>/<repo>.'.format(repo))
            continue

        if repo.lower() in ds['full_name'].str.lower().values:
            print('Dataset already contains \'{}\'.'.format(repo))
            continue

        print('Fetching info for \'{}\' repository...'.format(repo))
        org, name = parts[0], parts[1]

        metadata = get_repo_info(org, name, api_token)
        new_data.append(metadata)

    new_rows = pd.DataFrame(new_data)
    return pd.concat([ds, new_rows], ignore_index=True)


def main():
    parser = ArgumentParser(description='helper script for editting repo metadata dataset')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='repo csv file')
    parser.add_argument('-a', '--append', type=str, nargs='+', help='append repos to dataset. provide <org>/<repo>')
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    if args.append:
        nrows = df.shape[0]
        df = append(df, args.append)
        if df.shape[0] - nrows > 0:
            print('Writing dataset to \'{}\' with {} new row(s).'.format(args.dataset, df.shape[0] - nrows))
            df.to_csv(args.dataset, index=False, quoting=QUOTE_NONNUMERIC)


if __name__ == '__main__':
    main()

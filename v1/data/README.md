# Data

Scripts for collecting and processing the dataset.

## collect-repo-metadata.py and collect-repo-metadata.bash
The python script will query Github for repositories that match the particular query parameters.
It may take a while to run as it has to query each combination separately and sleep between them
to avoid rate-limiting from the GitHub API.
Upon running it will request your GitHub personal access token.
The resulting repo info is dumped into a csv specified by `--output`.

```
usage: collect-repo-metadata.py [-h] --tags TAGS [TAGS ...] [--languages LANGUAGES [LANGUAGES ...]] [--min-stars MIN_STARS] -o OUTPUT

Collects repos that meet tag, language, and star requirements.

optional arguments:
  -h, --help            show this help message and exit
  --tags TAGS [TAGS ...]
                        what tags to look for
  --languages LANGUAGES [LANGUAGES ...]
                        what languages to select repos for
  --min-stars MIN_STARS
                        minimum number of stars to consider
  -o OUTPUT, --output OUTPUT
                        where to output results
```

The bash script is just a wrapper around the python script that makes it easy to keep
the configuration consistent.

## edit-metadata.py
The helper script `edit-metadata.py` is there to help add repos to the db.
You can manually append repos as shown below.

```bash
python edit-metadata.py -d <csv_path> -a <repos...>

# for example
python edit-metadata.py -d repos-gt3.csv -a ceed/mfem ceed/laghos lanl/elements
```

## clone-repos.py
This script will take a set of repos specified in the csv outputted by `collect-repo-metadata.py` and 
clone them into some directory.
This also takes a while to run.
It will likely take up several GB of space and create a large number of INODES, so
set the root of the output to a filesystem that can handle this.

```
usage: clone-repos.py [-h] -d DATASET --root ROOT

Clone the repositories in a dataframe into a directory root.

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Where to find dataset of repos
  --root ROOT           Root directory to clone into
```

## collect-dataset.py
Collects the contents of all the source files and outputs them as a json lines file.

```
usage: collect-dataset.py [-h] --root ROOT -o OUTPUT

Create compact dataset representation

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           root to start searching for source files
  -o OUTPUT, --output OUTPUT
                        output path
```

## repo-plots.py and generate-all-repo-plots.bash
Create info plots about the source files (i.e. file extensions count, LOC, etc...).
The bash script is a helper to create all combinations of the plots.

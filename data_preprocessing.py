import pandas as pd
from surprise import Dataset, Reader

def load_movielens(path: str, dataset='100K'):
    """
    Load MovieLens data from a local path into a Surprise Dataset.
      - dataset: '100K' or '1M'
      - path: root folder, e.g., 'data/MovieLens_100K' or 'data/MovieLens_1M'
    """
    if dataset == '100K':
        ratings_file = f'{path}/u.data'
        sep = '\t'
    elif dataset == '1M':
        ratings_file = f'{path}/ratings.dat'
        sep = '::'
    else:
        raise ValueError("dataset must be '100K' or '1M'")

    # Read the raw file
    df = pd.read_csv(ratings_file,
                     sep=sep,
                     names=['userId', 'movieId', 'rating', 'timestamp'],
                     engine='python')
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader), df

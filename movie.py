#!/usr/bin/env python3
# save_movies_csv.py

import os
import pandas as pd

# ---- Configuration ----
DATA_PATH = 'ml-100k'    # adjust if your data lives elsewhere
INPUT_FILE = os.path.join(DATA_PATH, 'u.item')
OUTPUT_FILE = 'movies.csv'

def main():
    # Read only the columns we care about: movieId, title, IMDb_URL
    movies = pd.read_csv(
        INPUT_FILE,
        sep='|',
        encoding='latin-1',
        names=[
            'movieId','title','release_date','video_release_date','IMDb_URL',
            'unknown','Action','Adventure','Animation','Children','Comedy','Crime',
            'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery',
            'Romance','Sci-Fi','Thriller','War','Western'
        ],
        usecols=[0, 1, 4]
    )

    # Save to CSV
    movies.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(movies)} records to '{OUTPUT_FILE}'.")

if __name__ == '__main__':
    main()

import os
import pickle
import requests
from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import tensorflow as tf
from surprise.model_selection import train_test_split
from surprise import Dataset, Reader
from collections import defaultdict

# ---- Configuration ----
DATA_PATH       = 'ml-100k'
TMDB_API_KEY    = '6b889538be6267ab72d2f68ee415acb6'
TMDB_SEARCH_URL = 'https://api.themoviedb.org/3/search/movie'
TMDB_IMG_BASE   = 'https://image.tmdb.org/t/p/w342'
LOCAL_POSTER_DIR = os.path.join('static', 'posters')

# ---- Load Data ----
df_raw = pd.read_csv(
    f"{DATA_PATH}/u.data", sep='\t', names=['userId','movieId','rating','ts']
)
movies = pd.read_csv(
    f"{DATA_PATH}/u.item", sep='|', encoding='latin-1',
    names=[
        'movieId','title','release_date','video_release_date','IMDb_URL',
        'unknown','Action','Adventure','Animation','Children','Comedy','Crime',
        'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery',
        'Romance','Sci-Fi','Thriller','War','Western'
    ], usecols=[0,1,4]
)

# Prepare Surprise dataset & split
df = df_raw[['userId','movieId','rating']]
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df, reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Load models
models = {
    'Popularity': pickle.load(open('models/popularity.pkl','rb')),
    'User-User Collaborative Filtering': pickle.load(open('models/usercf.pkl','rb')),
    'Item-Item Collaborative Filtering': pickle.load(open('models/itemcf.pkl','rb')),
    'SVD':         pickle.load(open('models/svd.pkl','rb')),
    'NCF-Deep':    tf.keras.models.load_model('models/ncf_deep.h5', compile=False)
}

# Helpers
def get_poster_url(movie_id, title):
    # 1) Try TMDB
    try:
        if TMDB_API_KEY:
            resp = requests.get(
                TMDB_SEARCH_URL,
                params={'api_key': TMDB_API_KEY, 'query': title},
                timeout=5
            )
            resp.raise_for_status()
            results = resp.json().get('results') or []
            if results and results[0].get('poster_path'):
                return TMDB_IMG_BASE + results[0]['poster_path']
    except Exception:
        pass

    # 2) Fallback to local directory
    local_filename = os.path.join(LOCAL_POSTER_DIR, f"{movie_id}.jpg")
    print(local_filename)
    #if os.path.exists(local_filename):
    return url_for('static', filename=f"posters/{movie_id}.jpg")

    # 3) Final placeholder
    return 'https://via.placeholder.com/342x513?text=No+Image'

def get_pop_recs(pop, user_id, n=10):
    return pop.recommend(user_id, df_raw, n)

def get_surprise_recs(algo, user_id, n=10):
    try:
        uid = algo.trainset.to_inner_uid(user_id)
    except ValueError:
        return []
    seen = {iid for (iid, _) in algo.trainset.ur[uid]}
    candidates = set(range(algo.trainset.n_items)) - seen
    preds = [algo.predict(user_id, algo.trainset.to_raw_iid(i)) for i in candidates]
    preds.sort(key=lambda p: p.est, reverse=True)
    return [(int(p.iid), p.est) for p in preds[:n]]

def get_deep_recs(model, user_id, n=10):
    seen = set(df_raw[df_raw.userId==user_id].movieId)
    candidates = [m for m in movies.movieId.unique() if m not in seen]
    preds = [(m, model.predict([np.array([user_id]), np.array([m])]).flatten()[0]) for m in candidates]
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:n]

# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    recs = []
    selected = None
    if request.method == 'POST':
        user_id  = int(request.form['user_id'])
        selected = request.form['model']
        func = {
            'Popularity':    get_pop_recs,
            'User-User Collaborative Filtering':  get_surprise_recs,
            'Item-Item Collaborative Filtering':  get_surprise_recs,
            'SVD':           get_surprise_recs,
            'NCF-Deep':      get_deep_recs
        }[selected]
        raw_recs = func(models[selected], user_id)
        for mid, est in raw_recs:
            row = movies.loc[movies.movieId == mid].iloc[0]
            recs.append({
                'title':  row.title,
                'imdb':   row.IMDb_URL,
                'est':    round(est, 2),
                'poster': get_poster_url(mid, row.title)
            })
    return render_template(
        'index.html',
        model_names=list(models.keys()),
        recs=recs,
        selected=selected
    )

if __name__ == '__main__':
    app.run(debug=True)

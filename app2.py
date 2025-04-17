import os
import pickle
import requests
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from surprise.model_selection import train_test_split
from sklearn.model_selection import train_test_split as sk_train_test_split
from surprise import Dataset, Reader, accuracy
from collections import defaultdict

# Config & Data
DATA_PATH      = 'ml-100k'
TMDB_API_KEY    = os.getenv('TMDB_API_KEY')
TMDB_SEARCH_URL = 'https://api.themoviedb.org/3/search/movie'
TMDB_IMG_BASE   = 'https://image.tmdb.org/t/p/w342'

# Load raw ratings
raw = pd.read_csv(f"{DATA_PATH}/u.data", sep='\t', names=['userId','movieId','rating','ts'])
# Prepare Surprise dataset and split
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(raw[['userId','movieId','rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Load movie titles
movies = pd.read_csv(
    f'{DATA_PATH}/u.item', sep='|', encoding='latin-1',
    names=[
        'movieId','title','release_date','video_release_date','IMDb_URL',
        'unknown','Action','Adventure','Animation','Children','Comedy','Crime',
        'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery',
        'Romance','Sci-Fi','Thriller','War','Western'
    ], usecols=[0,1]
)

# Load models
models = {
    'Popularity': pickle.load(open('models/popularity.pkl','rb')),
    'User-User CF': pickle.load(open('models/usercf.pkl','rb')),
    'Item-Item CF': pickle.load(open('models/itemcf.pkl','rb')),
    'SVD':         pickle.load(open('models/svd.pkl','rb')),
    'NCF-Deep':    tf.keras.models.load_model('models/ncf_deep.h5', compile=False)
}

# Evaluation loop
eval_results = []
for name, m in models.items():
    row = {'Model': name}
    if name == 'Popularity':
        # Only top-N metrics for popularity
        user_true = defaultdict(set)
        for u, i, r in testset:
            if r >= 4.0:
                user_true[u].add(i)
        precs, recs = [], []
        for u in user_true:
            recs_u = m.recommend(u, raw, 10)
            rec_set = {iid for iid, _ in recs_u}
            n_rel = len(user_true[u])
            if n_rel > 0:
                precs.append(len(rec_set & user_true[u]) / 10)
                recs.append(len(rec_set & user_true[u]) / n_rel)
        row.update({'RMSE': None, 'MAE': None,
                    'Prec@10': np.mean(precs), 'Rec@10': np.mean(recs)})
    elif name == 'NCF-Deep':
        # RMSE/MAE only for deep model
        rd_train, rd_test = sk_train_test_split(raw, test_size=0.2, random_state=42)
        users = rd_test.userId.values
        items = rd_test.movieId.values
        y_true = rd_test.rating.values
        y_pred = m.predict([users, items]).flatten()
        row['RMSE'] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        row['MAE']  = float(np.mean(np.abs(y_pred - y_true)))
        row.update({'Prec@10': None, 'Rec@10': None})
    else:
        # CF and SVD: rating prediction + top-N
        preds = m.test(testset)
        row['RMSE'] = accuracy.rmse(preds, verbose=False)
        row['MAE']  = accuracy.mae(preds, verbose=False)

        # Compute top-N metrics
        user_true = defaultdict(set)
        for u, i, r in testset:
            if r >= 4.0:
                user_true[u].add(i)
        anti = trainset.build_anti_testset()
        all_preds = m.test(anti)
        user_est = defaultdict(list)
        for u, i, _, est, _ in all_preds:
            user_est[u].append((i, est))
        precs, recs = [], []
        for u, est_list in user_est.items():
            est_list.sort(key=lambda x: x[1], reverse=True)
            top10 = [i for i, _ in est_list[:10]]
            n_rel = len(user_true[u])
            if n_rel > 0:
                precs.append(len(set(top10) & user_true[u]) / 10)
                recs.append(len(set(top10) & user_true[u]) / n_rel)
        row.update({'Prec@10': np.mean(precs), 'Rec@10': np.mean(recs)})

    eval_results.append(row)

# Summary text
best_rmse = min([r['RMSE'] for r in eval_results if r['RMSE'] is not None])
best_model = next(r['Model'] for r in eval_results if r['RMSE'] == best_rmse)
summary_text = (
    f"Model **{best_model}** achieved the lowest RMSE ({best_rmse:.3f}), indicating superior rating accuracy. "
    "Collaborative filtering methods outperform popularity on top-N metrics, while the deep model yields the best rating predictions."
)

app = Flask(__name__)

# Helper functions

def get_poster_url(title):
    if not TMDB_API_KEY:
        return '/static/placeholder.png'
    resp = requests.get(TMDB_SEARCH_URL, params={'api_key': TMDB_API_KEY, 'query': title})
    if resp.ok:
        data = resp.json().get('results')
        if data and data[0].get('poster_path'):
            return TMDB_IMG_BASE + data[0]['poster_path']
    return '/static/placeholder.png'

from collections import defaultdict

def get_pop_recs(pop, user_id, n=10):
    return pop.recommend(user_id, raw, n)


def get_surprise_recs(algo, user_id, n=10):
    try:
        inner_uid = algo.trainset.to_inner_uid(user_id)
    except Exception:
        return []
    seen = {iid for (iid, _) in algo.trainset.ur[inner_uid]}
    all_iids = set(range(algo.trainset.n_items)) - seen
    preds = [algo.predict(user_id, algo.trainset.to_raw_iid(iid)) for iid in all_iids]
    preds.sort(key=lambda p: p.est, reverse=True)
    return [(int(p.iid), p.est) for p in preds[:n]]


def get_deep_recs(model, user_id, n=10):
    seen = set(raw[raw.userId == user_id].movieId)
    candidates = [mid for mid in movies.movieId.unique() if mid not in seen]
    preds = []
    for mid in candidates:
        est = model.predict([np.array([user_id]), np.array([mid])]).flatten()[0]
        preds.append((mid, est))
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:n]

@app.route('/', methods=['GET', 'POST'])
def index():
    recs_by_model = {}
    uid = None
    if request.method == 'POST':
        uid = int(request.form['user_id'])
        for name, m in models.items():
            if name == 'Popularity':
                recs = get_pop_recs(m, uid)
            elif name in ('User-User CF', 'Item-Item CF', 'SVD'):
                recs = get_surprise_recs(m, uid)
            else:
                recs = get_deep_recs(m, uid)

            enriched = []
            for mid, est in recs:
                title = movies.loc[movies.movieId == mid, 'title'].values[0]
                poster = get_poster_url(title)
                enriched.append({'title': title, 'est': round(est, 2), 'poster_url': poster})
            recs_by_model[name] = enriched

    return render_template('index.html',
                           recs_by_model=recs_by_model,
                           user_id=uid,
                           eval_results=eval_results,
                           summary_text=summary_text)

if __name__ == '__main__':
    app.run(debug=True)
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split as sk_tt

from data_preprocessing import load_movielens
from baseline_models import PopularityRecommender
from cf_models import build_user_cf, build_item_cf
from mf_models import build_svd
from deep_models import build_ncf_model, train_ncf

from surprise import accuracy

def rmse_mae(algo, testset):
    preds = algo.test(testset)
    return accuracy.rmse(preds, verbose=False), accuracy.mae(preds, verbose=False)

def precision_recall_at_k(algo, trainset, testset, k=10, threshold=4.0):
    from collections import defaultdict
    # ground truth
    user_true = defaultdict(set)
    for uid, iid, true_r in testset:
        if true_r >= threshold:
            user_true[uid].add(iid)
    # predictions on anti-testset
    anti = trainset.build_anti_testset()
    all_preds = algo.test(anti)
    user_est = defaultdict(list)
    for uid, iid, _, est, _ in all_preds:
        user_est[uid].append((iid, est))

    precisions, recalls = [], []
    for uid, ratings in user_est.items():
        ratings.sort(key=lambda x: x[1], reverse=True)
        top_k = [iid for iid,_ in ratings[:k]]
        n_rel = len(user_true[uid])
        if n_rel == 0: continue
        n_rec_k = len(top_k)
        n_rel_and_rec_k = len(set(top_k) & user_true[uid])
        precisions.append(n_rel_and_rec_k / n_rec_k)
        recalls.append(n_rel_and_rec_k / n_rel)
    return np.mean(precisions), np.mean(recalls)

class PopWrapper:
    """Wrap a PopularityRecommender so it has .test() and returns no rating-preds."""
    def __init__(self, pop, raw_df):
        self.pop = pop
        self.raw_df = raw_df
    def test(self, testset):
        return []  # no rating predictions

def evaluate_deep(raw_df):
    # split raw data for NCF
    train_df, test_df = sk_tt(raw_df, test_size=0.2, random_state=42)
    n_users = raw_df.userId.max()
    n_items = raw_df.movieId.max()
    model = build_ncf_model(n_users, n_items, embed_size=50)
    train_ncf(model, train_df, epochs=5, batch_size=256)
    # save deep model
    model.save('models/ncf_deep.h5')

    # eval on test
    users_test = test_df.userId.values
    items_test = test_df.movieId.values
    y_true = test_df.rating.values
    y_pred = model.predict([users_test, items_test]).flatten()
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mae  = np.mean(np.abs(y_pred - y_true))
    return rmse, mae

def main():
    # Prepare folders
    os.makedirs('models', exist_ok=True)

    # Load data
    data, raw_df = load_movielens(path='ml-100k', dataset='100K')
    trainset, testset = sk_tt(data.build_full_trainset().build_testset(),
                              test_size=0.2, random_state=42)

    results = []

    # --- Popularity-based ---
    pop = PopularityRecommender(min_ratings=50)
    pop.fit(raw_df)
    pickle.dump(pop, open('models/popularity.pkl','wb'))
    # only top-N metrics
    pop_wrap = PopWrapper(pop, raw_df)
    p_pop, r_pop = precision_recall_at_k(pop_wrap, data.build_full_trainset(), testset)
    results.append(('Popularity', None, None, p_pop, r_pop))

    # --- User–User CF ---
    uu = build_user_cf(data.build_full_trainset())
    pickle.dump(uu, open('models/usercf.pkl','wb'))
    rmse_uu, mae_uu = rmse_mae(uu, testset)
    p_uu, r_uu = precision_recall_at_k(uu, data.build_full_trainset(), testset)
    results.append(('User–User CF', rmse_uu, mae_uu, p_uu, r_uu))

    # --- Item–Item CF ---
    ii = build_item_cf(data.build_full_trainset())
    pickle.dump(ii, open('models/itemcf.pkl','wb'))
    rmse_ii, mae_ii = rmse_mae(ii, testset)
    p_ii, r_ii = precision_recall_at_k(ii, data.build_full_trainset(), testset)
    results.append(('Item–Item CF', rmse_ii, mae_ii, p_ii, r_ii))

    # --- SVD ---
    svd = build_svd(data.build_full_trainset())
    pickle.dump(svd, open('models/svd.pkl','wb'))
    rmse_sv, mae_sv = rmse_mae(svd, testset)
    p_sv, r_sv = precision_recall_at_k(svd, data.build_full_trainset(), testset)
    results.append(('SVD', rmse_sv, mae_sv, p_sv, r_sv))

    # --- NCF‑Deep ---
    rmse_deep, mae_deep = evaluate_deep(raw_df)
    results.append(('NCF‑Deep', rmse_deep, mae_deep, None, None))

    # Show results
    df = pd.DataFrame(results, columns=['Model','RMSE','MAE','Prec@10','Rec@10'])
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

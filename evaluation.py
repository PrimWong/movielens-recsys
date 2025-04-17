import numpy as np
from sklearn.model_selection import train_test_split as sk_tt
from deep_models import build_ncf_model, train_ncf

def evaluate_deep(raw_df):
    # split raw ratings into train/test
    train_df, test_df = sk_tt(raw_df, test_size=0.2, random_state=42)

    # build & train model
    n_users = raw_df.userId.max()
    n_items = raw_df.movieId.max()
    ncf = build_ncf_model(n_users, n_items, embed_size=50)
    train_ncf(ncf, train_df, epochs=5, batch_size=256)

    # predict on test set
    users_test = test_df.userId.values
    items_test = test_df.movieId.values
    y_true = test_df.rating.values
    y_pred = ncf.predict([users_test, items_test]).flatten()

    # compute RMSE/MAE
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mae  = np.mean(np.abs(y_pred - y_true))

    return rmse, mae

# … in your main evaluation_all(…) after SVD block …
rmse_deep, mae_deep = evaluate_deep(raw_df)
results.append(('NCF‑Deep', rmse_deep, mae_deep, np.nan, np.nan))

# baseline_models.py
import pandas as pd

class PopularityRecommender:
    """
    Simple popularity-based recommender:
     - rank items by average rating (optionally require a minimum count).
    """
    def __init__(self, min_ratings=0):
        self.min_ratings = min_ratings
        self.pop_scores = None

    def fit(self, ratings_df: pd.DataFrame):
        # ratings_df: columns ['userId','movieId','rating']
        grp = ratings_df.groupby('movieId')['rating']
        counts = grp.count()
        means  = grp.mean()
        # filter low‑count items if desired
        if self.min_ratings > 0:
            means = means[counts >= self.min_ratings]
        self.pop_scores = means.sort_values(ascending=False)

    def recommend(self, user_id, movies_df, n=10):
        """
        Return top-n popular movies, ignoring what the user has already rated.
        """
        # find what user has rated
        seen = set(movies_df[movies_df.userId == user_id].movieId)
        # filter out seen
        recs = [(iid, score) for iid, score in self.pop_scores.items() if iid not in seen]
        return recs[:n]

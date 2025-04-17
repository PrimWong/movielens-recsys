from surprise import KNNBasic

def build_user_cf(trainset, sim_options=None):
    """
    User–User Collaborative Filtering.
    """
    if sim_options is None:
        sim_options = {'name': 'cosine', 'user_based': True}
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    return algo

def build_item_cf(trainset, sim_options=None):
    """
    Item–Item Collaborative Filtering.
    """
    if sim_options is None:
        sim_options = {'name': 'cosine', 'user_based': False}
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    return algo

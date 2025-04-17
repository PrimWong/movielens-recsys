from surprise import SVD

def build_svd(trainset, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
    """
    Matrix Factorization via SVD.
    """
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs,
               lr_all=lr_all, reg_all=reg_all)
    algo.fit(trainset)
    return algo

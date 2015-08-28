"""
=====================================================================
Fusion of user ratings data, movie genres and data about movie actors
=====================================================================

This example demonstrates matrix completion on user-movie preference
data.

DISCLAIMER: Results can be improved if ideas from collaborative
filtering (regularization, user bias, item bias, etc.) are applied.
"""
print(__doc__)

import numpy as np

from skfusion import datasets
from skfusion import fusion as skf


def load_data():
    ratings_data, movies_data, actors_data = datasets.load_movielens()
    n_movies = 1000
    n_actors = 1000

    movies = set(m for val in ratings_data.values() for m in val)
    movies = list(movies)[:n_movies]
    movie2id = {m: i for i, m, in enumerate(movies)}
    user2id = {u: i for i, u in enumerate(ratings_data.keys())}
    genres = set(e for val1 in movies_data.values() for e in val1)
    genre2id = {g: i for i, g in enumerate(genres)}
    actors = set(e for m, val1 in actors_data.items()
                 for e in val1 if m in movie2id)
    actors = list(actors)[:n_actors]
    actor2id = {a: i for i, a in enumerate(actors)}

    R12_true = -1 * np.ones((len(user2id), len(movie2id)))
    print('User ratings: {}'.format(str(R12_true.shape)))
    for user in ratings_data:
        for movie in ratings_data[user]:
            if movie not in movie2id: continue
            R12_true[user2id[user], movie2id[movie]] = ratings_data[user][movie]
    R12_true = np.ma.masked_equal(R12_true, -1)
    R12_true = scale(R12_true, 0, 1)

    R23 = np.zeros((len(movie2id), len(genre2id)))
    print('Movie genres: {}'.format(str(R23.shape)))
    for movie in movies_data:
        for genre in movies_data[movie]:
            if movie not in movie2id: continue
            R23[movie2id[movie], genre2id[genre]] = 1.

    R24 = np.zeros((len(movie2id), len(actor2id)))
    print('Actors played: {}'.format(str(R24.shape)))
    for movie in actors_data:
        for actor in actors_data[movie]:
            if actor not in actor2id: continue
            if movie not in movie2id: continue
            R24[movie2id[movie], actor2id[actor]] = 1.

    frac = 0.9
    R12 = R12_true.copy()
    hide = np.logical_and(np.random.random(R12_true.shape) > frac, ~R12_true.mask)
    R12 = np.ma.masked_where(hide, R12)

    mean_user = np.mean(R12, 1)
    mean_movie = np.mean(R12, 0)
    mean_rating = np.mean(R12)
    means = mean_user, mean_movie, mean_rating

    p = 0.05
    t1 = skf.ObjectType('User', max(int(p*R12.shape[0]), 5))
    t2 = skf.ObjectType('Movie', max(int(p*R12.shape[1]), 5))
    t3 = skf.ObjectType('Genre', max(int(p*R23.shape[1]), 5))
    t4 = skf.ObjectType('Actor', max(int(p*R24.shape[1]), 5))

    # L2 regularization can be added via diagonal constraint matrices
    relations = [skf.Relation(R12, t1, t2, name='User ratings'),
                 skf.Relation(R23, t2, t3, name='Movie genres'),
                 skf.Relation(R24, t2, t4, name='Movie actors')]
    graph = skf.FusionGraph(relations)
    print('Ranks:', ''.join(['\n{}: {}'.format(o.name, o.rank)
                             for o in graph.object_types]))

    graph_small = skf.FusionGraph([skf.Relation(R12, t1, t2, name='User ratings')])

    return R12_true, hide, means, graph, graph_small


def rmse(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred)**2) / y_true.size)


def scale(X, amin, amax):
    return (X - X.min()) / (X.max() - X.min()) * (amax - amin) + amin


def main():
    R12_true, hidden, means, graph, graph_small = load_data()
    mean_user, mean_movie, mean_rating = means
    n_users, n_movies = R12_true.shape

    # mean rating
    score = rmse(R12_true[hidden], mean_rating)
    print('RMSE(mean rating): {}'.format(score))

    # mean user
    R12_pred = np.tile(mean_user.reshape((n_users, 1)), (1, n_movies))
    score = rmse(R12_true[hidden], R12_pred[hidden])
    print('RMSE(mean user): {}'.format(score))

    # mean movie
    R12_pred = np.tile(mean_movie.reshape((1, n_movies)), (n_users, 1))
    score = rmse(R12_true[hidden], R12_pred[hidden])
    print('RMSE(mean movie): {}'.format(score))

    # DFMF on ratings data only (it benefits if unknown values are set to mean)
    scores = []
    for _ in range(10):
        dfmf_fuser = skf.Dfmf(max_iter=100, init_type='random')
        dfmf_mod = dfmf_fuser.fuse(graph_small)
        R12_pred = dfmf_mod.complete(graph_small['User ratings'])
        # R12_pred = scale(R12_pred, 0, 1)
        R12_pred += np.tile(mean_user.reshape((n_users, 1)), (1, n_movies))
        R12_pred += np.tile(mean_movie.reshape((1, n_movies)), (n_users, 1))
        R12_pred = scale(R12_pred, 0, 1)
        scores.append(rmse(R12_true[hidden], R12_pred[hidden]))
    print('RMSE(ratings; out-of-sample dfmf): {}'.format(np.mean(scores)))

    # DFMF (it benefits if unknown values are set to mean)
    scores = []
    for _ in range(10):
        dfmf_fuser = skf.Dfmf(max_iter=100, init_type='random')
        dfmf_mod = dfmf_fuser.fuse(graph)
        R12_pred = dfmf_mod.complete(graph['User ratings'])
        # R12_pred = scale(R12_pred, 0, 1)
        R12_pred += np.tile(mean_user.reshape((n_users, 1)), (1, n_movies))
        R12_pred += np.tile(mean_movie.reshape((1, n_movies)), (n_users, 1))
        R12_pred = scale(R12_pred, 0, 1)
        scores.append(rmse(R12_true[hidden], R12_pred[hidden]))
    print('RMSE(ratings; out-of-sample dfmf): {}'.format(np.mean(scores)))

    # DFMC on ratings data only (proper treatment of unknown values)
    scores = []
    for _ in range(10):
        dfmc_fuser = skf.Dfmc(max_iter=100, init_type='random')
        dfmc_mod = dfmc_fuser.fuse(graph_small)
        R12_pred = dfmc_mod.complete(graph_small['User ratings'])
        R12_pred = scale(R12_pred, 0, 1)
        R12_pred += np.tile(mean_user.reshape((n_users, 1)), (1, n_movies))
        R12_pred += np.tile(mean_movie.reshape((1, n_movies)), (n_users, 1))
        R12_pred = scale(R12_pred, 0, 1)
        scores.append(rmse(R12_true[hidden], R12_pred[hidden]))
    print('RMSE(ratings; out-of-sample dfmc): {}'.format(np.mean(scores)))

    # DFMC (proper treatment of unknown values)
    scores = []
    for _ in range(10):
        dfmc_fuser = skf.Dfmc(max_iter=100, init_type='random')
        dfmc_mod = dfmc_fuser.fuse(graph)
        R12_pred = dfmc_mod.complete(graph['User ratings'])
        R12_pred = scale(R12_pred, 0, 1)
        R12_pred += np.tile(mean_user.reshape((n_users, 1)), (1, n_movies))
        R12_pred += np.tile(mean_movie.reshape((1, n_movies)), (n_users, 1))
        R12_pred = scale(R12_pred, 0, 1)
        scores.append(rmse(R12_true[hidden], R12_pred[hidden]))
    print('RMSE(ratings; out-of-sample dfmc): {}'.format(np.mean(scores)))

    # in-sample error (should be very close to zero for large rank values)
    score = rmse(R12_true[~hidden], R12_pred[~hidden])
    print('RMSE(in-sample dfmc): {}'.format(score))

    # PCA (requires scikit-learn module)
    from sklearn.decomposition import RandomizedPCA
    model = RandomizedPCA(n_components=10)
    R12 = graph['User ratings'].data.filled()
    pca_mod = model.fit(R12)
    R12_pred = scale(pca_mod.inverse_transform(pca_mod.transform(R12)), 0, 1)
    score = rmse(R12_true[hidden], R12_pred[hidden])
    print('RMSE(pca): {}'.format(score))

    # NMF (requires Nimfa module)
    import nimfa
    R12 = graph['User ratings'].data.filled()
    R12[R12 < 0] = 0
    lsnmf = nimfa.Lsnmf(R12, seed='random_vcol', rank=50, max_iter=100)
    lsnmf_fit = lsnmf()
    R12_pred = scale(np.dot(lsnmf_fit.basis(), lsnmf_fit.coef()), 0, 1)
    score = rmse(R12_true[hidden], R12_pred[hidden])
    print('RMSE(nmf): {}'.format(score))


if __name__ == "__main__":
    main()

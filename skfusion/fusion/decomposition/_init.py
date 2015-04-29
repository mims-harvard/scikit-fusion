from operator import itemgetter

import numpy as np


def initialize(obj_types, obj_type2n_obj, obj_type2rank, R, init_typ, random_state):
    init_types = {"random": _random, "random_c": _random_c, "random_vcol": _random_vcol}
    return init_types[init_typ](obj_types, obj_type2n_obj, obj_type2rank, R, random_state)


def _random(obj_types, obj_type2n_obj, obj_type2rank, R, random_state):
    G = {}
    for obj_type in obj_types:
        ni = obj_type2n_obj[obj_type]
        ci = obj_type2rank[obj_type]
        G[obj_type, obj_type] = random_state.rand(ni, ci)
    return G


def _random_c(obj_types, obj_type2n_obj, obj_type2rank, R, random_state):
    G = {}
    for obj_type in obj_types:
        ci = obj_type2rank[obj_type]
        G[obj_type, obj_type] = 1e-5 * np.ones((obj_type2n_obj[obj_type], ci))

        for obj_types, R12 in R.items():
            if obj_type not in obj_types:
                continue
            Rij = R12 if obj_type == obj_types[0] else R12.T
            p_c = int(.2 * Rij.shape[1])
            l_c = int(.5 * Rij.shape[1])
            cols_norm = [np.linalg.norm(Rij[:,i], 2) for i in range(Rij.shape[1])]
            top_c = sorted(enumerate(cols_norm), key=itemgetter(1), reverse=True)[:l_c]
            top_c = list(list(zip(*top_c))[0])
            Gi = np.zeros(G[obj_type, obj_type].shape)
            for i in range(ci):
                random_state.shuffle(top_c)
                Gi[:,i] = Rij[:, top_c[:p_c]].mean(axis=1)
            G[obj_type, obj_type] += np.abs(Gi)

    return G


def _random_vcol(obj_types, obj_type2n_obj, obj_type2rank, R, random_state):
    G = {}
    for obj_type in obj_types:
        ci = obj_type2rank[obj_type]
        G[obj_type, obj_type] = 1e-5 * np.ones((obj_type2n_obj[obj_type], ci))

        for obj_types, R12 in R.items():
            if obj_type not in obj_types:
                continue
            Rij = R12 if obj_type == obj_types[0] else R12.T
            p_c = int(.2 * Rij.shape[1])
            Gi = np.zeros(G[obj_type, obj_type].shape)
            idx = np.arange(Rij.shape[1])
            for i in range(ci):
                random_state.shuffle(idx)
                Gi[:, i] = Rij[:, idx[:p_c]].mean(axis=1)
            G[obj_type, obj_type] += np.abs(Gi)
    return G

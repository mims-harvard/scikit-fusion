from itertools import product

import numpy as np
from joblib import Parallel, delayed

from .base import FusionFit
from .models import _dfmc


__all__ = ['Dfmc']


def parallel_dfmc_wrapper(**params):
    return _dfmc.dfmc(**params)


class Dfmc(FusionFit):
    """Data fusion by matrix completion.

    Parameters
    ----------
    fusion_graph :

    Attributes
    ---------
    fusion_graph :
    """
    def __init__(self, fusion_graph):
        super(Dfmc, self).__init__(fusion_graph)

    def fuse(self, max_iter=100, init_type='random_c', n_run=1,
             stopping=None, stopping_system=None, verbose=0,
             compute_err=False, callback=None, random_state=None,
             n_jobs=1):
        """Run data fusion completion algorithm.

        Parameters
        ----------
        max_iter :
        init_type :
        n_run :
        stopping :
        stopping_system :
        verbose :
        compute_err :
        callback :
        random_state :
        n_jobs :
        """
        self.max_iter = max_iter
        self.init_type = init_type
        self.n_run = n_run
        if isinstance(random_state, np.random.RandomState):
            random_state = random_state
        else:
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

        object_types = set([ot for ot in self.fusion_graph.object_types])
        object_type2rank = {ot: int(ot.rank) for ot in self.fusion_graph.object_types}

        R, T, M = {}, {}, {}
        for row_type, col_type in product(self.fusion_graph.object_types, repeat=2):
            for relation in self.fusion_graph.get(row_type, col_type):
                if relation.row_type != relation.col_type:
                    R[relation.row_type, relation.col_type] = R.get((
                        relation.row_type, relation.col_type), [])
                    R[relation.row_type, relation.col_type].append(relation.data)

                    M[relation.row_type, relation.col_type] = M.get((
                        relation.row_type, relation.col_type), [])
                    M[relation.row_type, relation.col_type].append(relation.mask)
                else:
                    T[relation.row_type, relation.col_type] = T.get((
                        relation.row_type, relation.col_type), [])
                    T[relation.row_type, relation.col_type].append(relation.data)

        parallelizer = Parallel(n_jobs=n_jobs, max_nbytes=1e3, verbose=verbose)
        task_iter = (delayed(parallel_dfmc_wrapper)(
            R=R, M=M, Theta=T, obj_types=object_types,
            obj_type2rank=object_type2rank, max_iter=self.max_iter,
            init_type=init_type, stopping=stopping, stopping_system=stopping_system,
            verbose=verbose, compute_err=compute_err, callback=callback,
            random_state=random_state, n_jobs=n_jobs)
                     for _ in range(self.n_run))
        entries = parallelizer(task_iter)

        for G, S in entries:
            for (object_type, _), factor in G.items():
                self.factors_[object_type].append(factor)

            for (row_type, col_type), backbones in S.items():
                for i, relation in enumerate(self.fusion_graph.get(row_type, col_type)):
                    self.backbones_[relation].append(backbones[i])
        return self

from itertools import product
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed

from ..base import FusionFit
from ._dfmc import dfmc


__all__ = ['Dfmc']


def parallel_dfmc_wrapper(**params):
    return dfmc(**params)


class Dfmc(FusionFit):
    """Data fusion by matrix completion.

    Attributes
    ---------
    fusion_graph :
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
    def __init__(self, max_iter=100, init_type='random_c', n_run=1,
                 stopping=None, stopping_system=None, verbose=0,
                 compute_err=False, callback=None,
                 random_state=None, n_jobs=1):
        super(Dfmc, self).__init__()
        self._set_params(vars())

    def fuse(self, fusion_graph):
        """Run data fusion completion algorithm.

        Parameters
        ----------
        fusion_graph :
        """
        self.fusion_graph = fusion_graph
        if not isinstance(self.random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(self.random_state)

        object_types = set([ot for ot in self.fusion_graph.object_types])
        object_type2rank = {ot: int(ot.rank) for ot in self.fusion_graph.object_types}

        R, T, M = {}, {}, {}
        for row_type, col_type in product(self.fusion_graph.object_types, repeat=2):
            for relation in self.fusion_graph.get_relations(row_type, col_type):
                filled_data = relation.filled()
                if relation.preprocessor:
                    preprocessed_data = relation.preprocessor(filled_data)
                else:
                    preprocessed_data = filled_data
                if np.ma.is_masked(preprocessed_data):
                    data = preprocessed_data.data
                    mask = preprocessed_data.mask
                else:
                    data = preprocessed_data
                    mask = None
                if relation.row_type != relation.col_type:
                    R[relation.row_type, relation.col_type] = R.get((
                        relation.row_type, relation.col_type), [])
                    R[relation.row_type, relation.col_type].append(data)

                    M[relation.row_type, relation.col_type] = M.get((
                        relation.row_type, relation.col_type), [])
                    M[relation.row_type, relation.col_type].append(mask)
                else:
                    T[relation.row_type, relation.col_type] = T.get((
                        relation.row_type, relation.col_type), [])
                    T[relation.row_type, relation.col_type].append(data)

        parallelizer = Parallel(n_jobs=self.n_jobs, max_nbytes=1e3, verbose=self.verbose)
        task_iter = (delayed(parallel_dfmc_wrapper)(
            R=R, M=M, Theta=T, obj_types=object_types,
            obj_type2rank=object_type2rank, max_iter=self.max_iter, init_type=self.init_type,
            stopping=self.stopping, stopping_system=self.stopping_system, verbose=self.verbose,
            compute_err=self.compute_err, callback=self.callback, random_state=self.random_state,
            n_jobs=self.n_jobs)
                     for _ in range(self.n_run))
        entries = parallelizer(task_iter)

        self.factors_ = defaultdict(list)
        self.backbones_ = defaultdict(list)
        for G, S in entries:
            for (object_type, _), factor in G.items():
                self.factors_[object_type].append(factor)

            for (row_type, col_type), backbones in S.items():
                for i, relation in enumerate(self.fusion_graph.get_relations(row_type, col_type)):
                    self.backbones_[relation].append(backbones[i])
        return self

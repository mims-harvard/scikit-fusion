from itertools import product
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed

from ..base import FusionFit, FusionTransform
from ._dfmf import dfmf, transform


__all__ = ['Dfmf', 'DfmfTransform']


def parallel_dfmf_wrapper(**params):
    return dfmf(**params)


class Dfmf(FusionFit):
    """Data fusion by matrix factorization.

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
                 compute_err=False, callback=None, random_state=None,
                 n_jobs=1):
        super(Dfmf, self).__init__()
        self._set_params(vars())

    def fuse(self, fusion_graph):
        """Run data fusion factorization algorithm.

        Parameters
        ----------
        fusion_graph :
        """
        self.fusion_graph = fusion_graph
        if not isinstance(self.random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(self.random_state)

        object_types = set([ot for ot in self.fusion_graph.object_types])
        object_type2rank = {ot: int(ot.rank) for ot in self.fusion_graph.object_types}

        R, T = {}, {}
        for row_type, col_type in product(self.fusion_graph.object_types, repeat=2):
            for relation in self.fusion_graph.get_relations(row_type, col_type):
                filled_data = relation.filled()
                if relation.preprocessor:
                    preprocessed_data = relation.preprocessor(filled_data)
                else:
                    preprocessed_data = filled_data

                if np.ma.is_masked(preprocessed_data):
                    data = preprocessed_data.data
                else:
                    data = preprocessed_data
                X = R if relation.row_type != relation.col_type else T
                X[relation.row_type, relation.col_type] = X.get((
                    relation.row_type, relation.col_type), [])
                X[relation.row_type, relation.col_type].append(data)

        parallelizer = Parallel(n_jobs=self.n_jobs, max_nbytes=1e3, verbose=self.verbose)
        task_iter = (delayed(parallel_dfmf_wrapper)(
            R=R, Theta=T, obj_types=object_types, obj_type2rank=object_type2rank,
            max_iter=self.max_iter, init_type=self.init_type, stopping=self.stopping,
            stopping_system=self.stopping_system, verbose=self.verbose,
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


def parallel_dfmf_transform_wrapper(fuser, run, **params):
    G = {(object_type, object_type): fuser.factor(object_type, run)
         for object_type in fuser.fusion_graph.object_types}
    S = {(relation.row_type, relation.col_type): [fuser.backbone(relation, run)]
         for relation in fuser.fusion_graph.relations
         if relation.row_type != relation.col_type}
    return transform(G=G, S=S, **params)


class DfmfTransform(FusionTransform):
    """Online data transformer into fused space.

    Attributes
    ----------
    target :
    fusion_graph :
    fuser :
    max_iter :
    init_type :
    stopping :
    stopping_system :
    fill_value :
    verbose :
    compute_err :
    random_state :
    n_jobs :

    Parameters
    ----------
    fuser :
    max_iter :
    init_type :
    stopping :
    stopping_system :
    fill_value :
    verbose :
    compute_err :
    random_state :
    n_jobs :
    """
    def __init__(self, max_iter=100, init_type=None, n_run=1, stopping=None,
                 stopping_system=None, fill_value=0, verbose=0, compute_err=False,
                 callback=None, random_state=None, n_jobs=1):
        super(DfmfTransform, self).__init__()
        self._set_params(vars())

    def transform(self, target, fusion_graph, fuser):
        """Transform the data into the space given by Fuser.

        Parameters
        ----------
        target :
        fusion_graph :
        fuser :
        """
        self.target = target
        self.fusion_graph = fusion_graph
        self.fuser = fuser
        self._validate_graph()

        init_type = self.init_type if self.init_type is not None else self.fuser.init_type
        if not isinstance(self.random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(self.random_state)

        object_type2rank = {ot: int(ot.rank) for ot in self.fusion_graph.object_types}

        R, T = {}, {}
        for row_type, col_type in product(self.fusion_graph.object_types, repeat=2):
            for relation in self.fusion_graph.get_relations(row_type, col_type):
                if relation.preprocessor:
                    data = relation.preprocessor(relation.data)
                else:
                    data = relation.data
                if np.ma.is_masked(data):
                    data.fill_value = self.fill_value
                    data = data.filled()
                data[~np.isfinite(data)] = self.fill_value
                X = R if relation.row_type != relation.col_type else T
                X[relation.row_type, relation.col_type] = X.get((
                    relation.row_type, relation.col_type), [])
                X[relation.row_type, relation.col_type].append(data)

        parallelizer = Parallel(n_jobs=self.n_jobs, max_nbytes=1e3, verbose=self.verbose)
        task_iter = (delayed(parallel_dfmf_transform_wrapper)(
            self.fuser, run,
            R_ij=R, Theta_i=T, target_obj_type=self.target,
            obj_type2rank=object_type2rank, max_iter=self.max_iter, init_type=init_type,
            stopping=self.stopping, stopping_system=self.stopping_system, verbose=self.verbose,
            compute_err=self.compute_err, callback=self.callback, random_state=self.random_state)
                     for run in range(self.n_run))
        entries = parallelizer(task_iter)

        self.factors_ = defaultdict(list)
        for G_new in entries:
            self.factors_[self.target].append(G_new)
        return self

from itertools import product

import numpy as np

from .base import FusionFit, FusionTransform
from .models import _dfmf


__all__ = ['Dfmf', 'DfmfTransform']


class Dfmf(FusionFit):
    """Data fusion by matrix factorization.

    Parameters
    ----------
    fusion_graph :

    Attributes
    ---------
    fusion_graph :
    """
    def __init__(self, fusion_graph):
        super(Dfmf, self).__init__(fusion_graph)

    def fuse(self, max_iter=100, init_type='random_c', n_run=1,
             stopping=None, stopping_system=None, verbose=0,
             compute_err=False, callback=None, random_state=None, n_jobs=1):
        """Run data fusion factorization algorithm.

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

        R, T = {}, {}
        for row_type, col_type in product(self.fusion_graph.object_types, repeat=2):
            for relation in self.fusion_graph.get(row_type, col_type):
                X = R if relation.row_type != relation.col_type else T
                X[relation.row_type, relation.col_type] = X.get((
                    relation.row_type, relation.col_type), [])
                X[relation.row_type, relation.col_type].append(relation.data)

        for run in range(n_run):
            G, S = _dfmf.dfmf(
                R=R, Theta=T, obj_types=object_types,
                obj_type2rank=object_type2rank, max_iter=self.max_iter,
                init_type=init_type, stopping=stopping,
                stopping_system=stopping_system, verbose=verbose,
                compute_err=compute_err, callback=callback,
                random_state=self.random_state, n_jobs=n_jobs)

            for (object_type, _), factor in G.items():
                self.factors_[object_type].append(factor)

            for (row_type, col_type), backbones in S.items():
                for i, relation in enumerate(self.fusion_graph.get(row_type, col_type)):
                    self.backbones_[relation].append(backbones[i])
        return self


class DfmfTransform(FusionTransform):
    """Online data transformer into fused space.

    Parameters
    ----------
    target :
    fusion_graph :
    fuser :

    Attributes
    ----------
    target :
    fusion_graph :
    fuser :
    """
    def __init__(self, target, fusion_graph, fuser):
        super(DfmfTransform, self).__init__(target, fusion_graph, fuser)

    def transform(self, max_iter=None, init_type=None, stopping=None,
                  stopping_system=None, verbose=0, compute_err=False,
                  random_state=None):
        """Transform the data into the space given by Fuser.

        Parameters
        ----------
        fuser :
        max_iter :
        init_type :
        stopping :
        stopping_system :
        verbose :
        compute_err :
        random_state :
        """
        max_iter = max_iter if max_iter else self.fuser.max_iter
        init_type = init_type if init_type else self.fuser.init_type
        if isinstance(random_state, np.random.RandomState):
            random_state = random_state
        else:
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

        object_type2rank = {ot: int(ot.rank) for ot in self.fusion_graph.object_types}

        R, T = {}, {}
        for row_type, col_type in product(self.fusion_graph.object_types, repeat=2):
            for relation in self.fusion_graph.get(row_type, col_type):
                X = R if relation.row_type != relation.col_type else T
                X[relation.row_type, relation.col_type] = X.get((
                    relation.row_type, relation.col_type), [])
                X[relation.row_type, relation.col_type].append(relation.data)

        for run in range(self.fuser.n_run):
            G = {(object_type, object_type): self.fuser.factor(object_type, run)
                 for object_type in self.fuser.fusion_graph.object_types}
            S = {(relation.row_type, relation.col_type): [self.fuser.backbone(relation, run)]
                 for relation in self.fuser.fusion_graph.relations}

            G_new = _dfmf.transform(
                R_ij=R, Theta_i=T, target_obj_type=self.target,
                obj_type2rank=object_type2rank, G=G, S=S, max_iter=max_iter,
                init_type=init_type, stopping=stopping, stopping_system=stopping_system,
                verbose=verbose, compute_err=compute_err, random_state=self.random_state)

            self.factors_[self.target].append(G_new)
        return self

from collections import defaultdict
from abc import ABCMeta, abstractmethod

import numpy as np


__all__ = ['FusionBase', 'FusionFit', 'FusionTransform', 'DataFusionError']


class FusionBase(object):
    """Base class for data fusion.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Attributes
    ----------
    factors_ :
    backbones_ :
    """
    __metaclass__ = ABCMeta
    _params = None

    @abstractmethod
    def __init__(self):
        self.factors_ = defaultdict(list)
        self.backbones_ = defaultdict(list)

    def _set_params(self, values):
        self._params = values
        # first argument is 'self'
        del self._params['self']
        self.__dict__.update(self._params)

    def factor(self, object_type, run=None):
        """Return fused latent matrix factor of an object type.

        Parameters
        ----------
        object_type :
        run :

        Returns
        -------
        G :
        """
        if object_type not in self.fusion_graph.object_types:
            raise DataFusionError("Object type %s is not included "
                                  "in the fusion scheme" % object_type.name)
        if object_type not in self.factors_:
            raise DataFusionError("Unknown object type.")
        if self.n_run > 1 and run is None:
            return self._factor_iter(object_type)
        else:
            run = 0 if run is None else run
            return self.factors_[object_type][run]

    def _factor_iter(self, object_type):
        """Return an iterator over latent matrix factors from
        different runs.

        Parameters
        ----------
        object_type :
        """
        for run in range(self.n_run):
            yield self.factors_[object_type][run]

    def chain(self, row_type, col_type):
        """Express objects of type ``row_type`` in the fused latent space of
        ``col_type`` by propagating latent factors over the fusion graph.

        Parameters
        ----------
        row_type :
        col_type :

        Returns
        -------
        chains :
        """
        paths = [[row_type]]
        if row_type == col_type:
            yield paths[0]
        while paths:
            paths_new = []
            for path in paths:
                expand = [ot for ot in self.fusion_graph.out_neighbors(path[-1])
                          if ot not in path]
                refined_paths = [path + [ot] for ot in expand]
                for refined in refined_paths:
                    if refined[-1] == col_type:
                        yield refined
                    else:
                        paths_new.append(refined)
            paths = paths_new

    def __str__(self):
        pparams = ', '.join('{}={}'.format(k, v) for k, v in self._params.items())
        return '{}({})'.format(self.__class__.__name__, pparams)

    def __repr__(self):
        pparams = ', '.join('{}={}'.format(k, v) for k, v in self._params.items())
        return '{}({})'.format(self.__class__.__name__, pparams)


class FusionFit(FusionBase):
    """Base class for fused data space estimation.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        super(FusionFit, self).__init__()

    def complete(self, relation, run=None):
        """Return reconstructed relation matrix.

        Parameters
        ----------
        relation :
        run :

        Returns
        -------
        R12_hat : array-like, shape (n_obj1, n_obj2)
        """
        if relation.row_type not in self.fusion_graph.object_types or \
            relation.col_type not in self.fusion_graph.object_types:
            raise DataFusionError("Object type %s or %s are not included "
                                  "in the fusion scheme" % (
                relation.row_type.name, relation.col_type.name))
        if self.n_run > 1 and run is None:
            return self._complete_iter(relation)
        else:
            run = 0 if run is None else run
            G1 = self.factor(relation.row_type, run)
            S12 = self.backbone(relation, run)
            G2 = self.factor(relation.col_type, run)
            R12_hat = np.dot(G1, np.dot(S12, G2.T))
            if relation.postprocessor:
                R12_hat = relation.postprocessor(R12_hat)
            return R12_hat

    def _complete_iter(self, relation):
        """Return an iterator over completed relation matrix obtained
        from different runs

        Parameters
        ----------
        relation :
        run :

        Returns
        -------
        """
        for run in range(self.n_run):
            G1 = self.factor(relation.row_type, run)
            S12 = self.backbone(relation, run)
            G2 = self.factor(relation.col_type, run)
            R12_hat = np.dot(G1, np.dot(S12, G2.T))
            if relation.postprocessor:
                R12_hat = relation.postprocessor(R12_hat)
            yield R12_hat

    def backbone(self, relation, run=None):
        """Return fused latent matrix factor of a relation.

        Parameters
        ----------
        relation :
        run :

        Returns
        -------
        """
        if relation.row_type not in self.fusion_graph.object_types \
                or relation.col_type not in self.fusion_graph.object_types:
            raise DataFusionError('Object types are not recognized.')
        if relation not in self.backbones_:
            raise DataFusionError("Unknown relation.")
        if self.n_run > 1 and run is None:
            return self._backbone_iter(relation)
        else:
            run = 0 if run is None else run
            return self.backbones_[relation][run]

    def _backbone_iter(self, relation):
        """Return an iterator over backbone latent matrix factors from
        different runs.

        Parameters
        ----------
        relation :

        Returns
        -------
        """
        for run in range(self.n_run):
            yield self.backbones_[relation][run]


class FusionTransform(FusionBase):
    """Base class for online data transformation into the fused space

    Warning: This class should not be used directly.
    Use derived classes instead.

    Attributes
    ----------
    target :
    fusion_graph :
    fuser :
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        super(FusionTransform, self).__init__()

    def _validate_graph(self):
        if self.target not in self.fusion_graph.object_types:
            raise DataFusionError("Object type %s is not included " \
                                  "in the fusion scheme." % self.target.name)
        for relation in self.fusion_graph.relations:
            if self.target not in [relation.row_type, relation.col_type]:
                raise DataFusionError("Relation must include target "
                                  "object type: %s." % self.target.name)

    def chain(self, row_type=None, col_type=None):
        """Express objects of type ``target`` in the fused latent space of
        ``col_type`` by propagating latent factors over the fusion graph.

        Parameters
        ----------
        row_type :
        col_type :
        """
        if row_type is not None and col_type is not None and \
                        row_type is not self.target:
            raise DataFusionError("Starting type should be target type: "
                                  "%s" % self.target.name)
        col_type = row_type if col_type is None else col_type
        return FusionBase.chain(self, self.target, col_type)


class DataFusionError(Exception):
    pass

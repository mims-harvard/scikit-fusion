from abc import ABCMeta, abstractmethod

import numpy as np

from ._base.fusion import FusionBase, DataFusionError

__all__ = ['FusionFit', 'FusionTransform']


class FusionFit(FusionBase):
    """Base class for fused data space estimation.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Attributes
    ----------
    fusion_graph :
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, fusion_graph):
        super(FusionFit, self).__init__(fusion_graph)

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
            raise DataFusionError("Fused latent factors are not computed yet.")
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
    def __init__(self, target, fusion_graph, fuser):
        super(FusionTransform, self).__init__(fusion_graph)
        self.target = target
        self.fuser = fuser
        if self.target not in self.fusion_graph.object_types:
            raise DataFusionError("Object type %s is not included " \
                                  "in the fusion scheme." % self.target.name)
        self._validate_graph()

    def _validate_graph(self):
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

from collections import defaultdict
from abc import ABCMeta, abstractmethod

__all__ = ['FusionBase', 'DataFusionError']


class FusionBase(object):
    """Base class for data fusion.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Attributes
    ----------
    fusion_graph :
    factors_ :
    backbones_ :
    n_run :
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, fusion_graph):
        """
        Parameters
        ----------
        fusion_graph :
        """
        self.fusion_graph = fusion_graph
        self.factors_ = defaultdict(list)
        self.backbones_ = defaultdict(list)
        self.n_run = 1

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
            raise DataFusionError("Fused latent factors are not computed yet.")
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
                expand = [
                    ot for ot in self.fusion_graph.object_types
                    if ot in self.fusion_graph.adjacency_matrix[path[-1]]
                    and ot not in path]
                refined_paths = [path + [ot] for ot in expand]
                for refined in refined_paths:
                    if refined[-1] == col_type:
                        yield refined
                    else:
                        paths_new.append(refined)
            paths = paths_new


class DataFusionError(Exception):
    pass

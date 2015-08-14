#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict, OrderedDict, Iterable
from uuid import uuid1 as uuid
from numbers import Number

import numpy as np

from .base import DataFusionError


__all__ = ['FusionGraph', 'Relation', 'ObjectType']


class FusionGraph(object):
    """Container object for data sets and object types.

    Parameters
    ----------
    relations :

    Attributes
    ----------
    adjacency_matrix
    relations:
    object_types :
    """
    def __init__(self, relations=()):
        self.adjacency_matrix = {}
        self.relations = OrderedDict()
        self.object_types = OrderedDict()
        self._name2relation = {}
        self._name2object_type = {}
        self.add_relations_from(relations)

    @property
    def n_relations(self):
        return len(self.relations)

    @property
    def n_object_types(self):
        return len(self.object_types)

    def __getitem__(self, key):
        return self.adjacency_matrix.get(key, self._name2relation.get(key, None))

    def __setitem__(self, key, value):
        self.adjacency_matrix[key] = value

    def draw_networkx(self, filename=None, ax=None, *args, **kwargs):
        """Draw the data fusion graph using NetworkX and Matplotlib.

        Parameters
        ----------
        filename : str or file-like object
            A filename to output to. If str, the extension implies the format.
            If file-like object, pass the desired `format` explicitly.
            If None, the plot is drawn to a Matplotlib Axes object (can be
            supplied as `ax` keyword argument).

        **kwargs : optional keyword arguments
            Passed to ``networkx.draw_networkx()`` (and, optionally,
            ``matplotlib.figure.Figure.savefig()``).
        """
        import networkx as nx

        if filename and not ax:
            from matplotlib.figure import Figure
            ax = Figure().add_subplot(111)

        G = nx.MultiDiGraph()
        G.add_nodes_from(o.name for o in self.object_types)

        ot2count = defaultdict(int)
        for relation in self.relations:
            ot1 = relation.row_type
            ot2 = relation.col_type
            ot2count[ot1, ot2] += 1
            if ot1 != ot2:
                label = (r'$<\mathbf{R}_{%s,%s}^%d>$' %
                         (ot1.name, ot2.name, ot2count[ot1, ot2]))
            else:
                label = (r'$<\mathbf{\Theta}_%s^%d>$' %
                         (ot1.name, ot2count[ot1, ot2]))
            G.add_edge(ot1.name, ot2.name, label=label)

        nx.draw_networkx(G, *args,
                         ax=ax,
                         node_size=3000,
                         node_color='white',
                         **kwargs)
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, nx.spring_layout(G), edge_labels=edge_labels)

        if filename:
            ax.figure.savefig(filename, **kwargs)
        return G

    def draw_graphviz(self, *args, **kwargs):
        """Draw the data fusion graph using PyGraphviz and save it to a file.

        Parameters
        ----------
        graph_attr : dict
            Dict of Graphviz graph attributes
        node_attr : dict
            Dict of Graphviz node attributes
        edge_attr : dict
            Dict of Graphviz edge attributes

        *args, **kwargs : optional
            Passed to `pygraphviz.AGraph.draw()` method.
        """
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        # From http://graphviz.org/content/attrs
        G.graph_attr.update({
            'outputorder': 'edgesfirst',
            'packmode': 'graph',
            'pad': .3,
        }, **kwargs.pop('graph_attr', {}))
        G.node_attr.update({
            'fontsize': 11,
            'fontname': 'sans-serif',
            'fillcolor': 'white',
            'style': 'filled',
        }, **kwargs.pop('node_attr', {}))
        G.edge_attr.update({
            'fontsize': 9,
            'fontname': 'sans-serif',
        }, **kwargs.pop('edge_attr', {}))

        smallsize = .8 * float(G.node_attr['fontsize'])
        n_objects = {}
        for ot in self.object_types:
            # The maximum number of objects of this type featured in any of the
            # relations
            n = max(max([rel.data.shape[0] for rel in self.out_relations(ot)], default=0),
                    max([rel.data.shape[1] for rel in self.in_relations(ot)],  default=0))
            n_objects[ot] = n
            G.add_node(ot.name,
                       # This is relied upon by biolab/orange3; if you change this id,
                       # please let them know:
                       id='node `%s`' % ot.name,
                       label=('<%s<br/><font point-size="%.1f" color="grey">'
                               '%d</font>>' % (ot.name, smallsize, n)))
        relations = defaultdict(list)
        for rel in self.relations:
            relations[(rel.row_type, rel.col_type)].append(rel)
        for (ot1, ot2), rels in relations.items():
            label = (',<br/>&nbsp;'.join(rel.name for rel in rels if rel.name) or
                     '<b>%s</b>' % ('R' if ot1 != ot2 else '&Theta;'))
            label = '<&nbsp;' + label + '>'
            tooltip = ', '.join('[%d×%d]' % rel.data.shape for rel in rels)
            # Penwidth is normalized as the sum of relations' (defined) areas
            # divided by the largest possible area of the given two object types
            weight = sum(np.ma.count(rel.data) / n_objects[ot1] / n_objects[ot2]
                         for rel in rels)
            penwidth = np.clip(1.3 * weight, .5, 3)
            G.add_edge(ot1.name, ot2.name,
                       # This is relied upon by biolab/orange3; if you change this id,
                       # please let them know
                       id='edge `%s`->`%s`' % (ot1.name, ot2.name),
                       label=label,
                       tooltip=tooltip,
                       labelaligned=True,  # http://www.graphviz.org/content/allign-edge-labels-fit-its-path-svg-output
                       penwidth=penwidth)

        if len(args) < 3 and 'prog' not in kwargs:
            kwargs['prog'] = 'dot'
        G.draw(*args, **kwargs)

    def add_relation(self, relation):
        """Add a single relation to the fusion graph.

        Parameters
        ----------
        relation :
        """
        self.relations[relation] = True
        if relation.name:
            self._name2relation[relation.name] = relation
        self.object_types[relation.row_type] = True
        self.object_types[relation.col_type] = True
        self._name2object_type[relation.row_type.name] = relation.row_type
        self._name2object_type[relation.col_type.name] = relation.col_type
        neighbors = self.adjacency_matrix.get(relation.row_type, {})
        nbs_list = neighbors.get(relation.col_type, []) + [relation]
        neighbors[relation.col_type] = nbs_list
        self.adjacency_matrix[relation.row_type] = neighbors

    def add_relations_from(self, relations):
        """Add relations to the fusion graph.

        Parameters
        ----------
        relations : container of Relation-s
        """
        for relation in relations:
            self.add_relation(relation)

    def remove_relation(self, relation):
        """Remove a single relation from the fusion graph.

        Parameters
        ----------
        relation :
        """
        self.adjacency_matrix[relation.row_type][relation.col_type].remove(relation)
        self.relations.pop(relation)
        if relation.name:
            self._name2relation.pop(relation.name, None)
        if not self.adjacency_matrix[relation.row_type][relation.col_type]:
            self.adjacency_matrix[relation.row_type].pop(relation.col_type, None)
        if not list(self.in_neighbors(relation.row_type)) and \
                not list(self.out_neighbors(relation.row_type)):
            self.remove_object_type(relation.row_type)
            if relation.row_type == relation.col_type:
                return
        if not list(self.in_neighbors(relation.col_type)) and \
                not list(self.out_neighbors(relation.col_type)):
            self.remove_object_type(relation.col_type)

    def remove_relations_from(self, relations):
        """Remove relations from the fusion graph.

        Parameters
        ----------
        relations : container of Relation-s
        """
        for relation in relations:
            self.remove_relation(relation)

    def remove_object_type(self, object_type):
        """Remove a single relation from the fusion graph.

        Parameters
        ----------
        object_type :
        """
        for relation in self.relations:
            if object_type in relation:
                self.remove_relation(relation)
        self.adjacency_matrix.pop(object_type, None)
        for obj_type in self.adjacency_matrix:
            self.adjacency_matrix[obj_type].pop(object_type, None)
        self._name2object_type.pop(object_type.name, None)
        self.object_types.pop(object_type)

    def remove_object_types_from(self, object_types):
        """Remove relations from the fusion graph.

        Parameters
        ----------
        object_types: container of ObjectType-s
        """
        for object_type in object_types:
            self.remove_object_type(object_type)

    def get_relation(self, name):
        """Return a relation matrix with a given name.

        Parameters
        ----------
        name : str
            Name of the relation
        """
        if name not in self._name2relation:
            raise DataFusionError("Relation name unknown")
        return self._name2relation[name]

    def get_relations(self, row_type, col_type):
        """Return an iterator for relation matrices between two types of objects.

        Parameters
        ----------
        row_type : ObjectType
        col_type : ObjectType

        Returns
        -------
        relation :  iterator
        """
        if row_type not in self.object_types or col_type not in self.object_types:
            raise DataFusionError("Object types are not recognized.")
        return iter(self.adjacency_matrix.get(row_type, {}).get(col_type, []))

    def get_object_type(self, name):
        """Return object type whose name is provided.

        Parameters
        ----------
        name : str
            Name of the object type
        """
        if name not in self._name2object_type:
            raise DataFusionError("Object type name unknown")
        return self._name2object_type[name]

    def get_names(self, object_type):
        """Get names of all possible object type row/column names.

        Parameters
        ----------
        object_type : ObjectType

        Returns
        -------
        List of names when they exist, None otherwise.
        """
        if isinstance(object_type, str):
            object_type = self.get_object_type(object_type)

        size = 0
        for rel in self.out_relations(object_type):
            if rel.row_names:
                return rel.row_names
            else:
                size = rel.data.shape[0]

        for rel in self.in_relations(object_type):
            if rel.col_names:
                return rel.col_names
            else:
                size = rel.data.shape[1]

        return [str(x) for x in range(size)]

    def get_metadata(self, object_type):
        """Get metadata for given object type.

        Parameters
        ----------
        object_type : ObjectType

        Returns
        -------
        Metadata (list of dicts)
        """
        if isinstance(object_type, str):
            object_type = self.get_object_type(object_type)

        metadata = [{} for x in self.get_names(object_type)]

        for rel in self.out_relations(object_type):
            if rel.row_metadata:
                for md1, md2 in zip(metadata, rel.row_metadata):
                    md1.update(md2)

        for rel in self.in_relations(object_type):
            if rel.col_metadata:
                for md1, md2 in zip(metadata, rel.col_metadata):
                    md1.update(md2)
        return metadata


    def out_relations(self, object_type):
        """Return an iterator for relations adjacent to the object type.

        Parameters
        ----------
        object_type : ObjectType

        Returns
        -------
        relation : iterator
        """
        if object_type not in self.object_types:
            raise DataFusionError("Object type not in the fusion graph.")
        for col_type in self.adjacency_matrix.get(object_type, {}):
            for relation in self.adjacency_matrix[object_type][col_type]:
                yield relation

    def in_relations(self, object_type):
        """Return an iterator for relations adjacent to the object type.

        Parameters
        ----------
        object_type : ObjectType

        Returns
        -------
        relation : iterator
        """
        if object_type not in self.object_types:
            raise DataFusionError("Object type not in the fusion graph.")
        for row_type in self.adjacency_matrix:
            for relation in self.adjacency_matrix[row_type].get(object_type, {}):
                yield relation

    def out_neighbors(self, object_type):
        """Return an iterator for object types adjacent to the object type.

        Parameters
        ----------
        object_type : ObjectType

        Returns
        -------
        relation : iterator
        """
        if object_type not in self.object_types:
            raise DataFusionError("Object type not in the fusion graph.")
        return iter(self.adjacency_matrix.get(object_type, {}).keys())

    def in_neighbors(self, object_type):
        """Return an iterator for object types adjacent to the object type.

        Parameters
        ----------
        object_type : ObjectType

        Returns
        -------
        relation : iterator
        """
        if object_type not in self.object_types:
            raise DataFusionError("Object type not in the fusion graph.")
        for row_type in self.adjacency_matrix.keys():
            if object_type in self.adjacency_matrix[row_type]:
                if len(self.adjacency_matrix[row_type][object_type]) > 0:
                    yield row_type

    def __str__(self):
        return "{}(Object types: {}, Relations: {})".format(
            self.__class__.__name__, len(self.object_types), len(self.relations))

    def __repr__(self):
        return "{}(Object types={}, Relations={})".format(
            self.__class__.__name__,
            repr(list(self.object_types.keys())),
            repr(list(self.relations.keys())))


class ObjectType(object):
    """Object type used for fusion.

    Attributes
    ----------
    name :
    rank :
    """
    def __init__(self, name, rank=5):
        self.name = name
        self.rank = rank

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.name

    def __repr__(self):
        return '{}("{}")'.format(self.__class__.__name__, self.name)


def fill_mean(x):
    mean = np.nanmean(x)
    if np.ma.is_masked(x):
        indices = np.logical_or(~np.isfinite(x), x.mask)
    else:
        indices = ~np.isfinite(x)
    filled = x.copy()
    filled[indices] = mean
    return filled


def fill_row(x):
    row_mean = np.nanmean(x, 1)
    mat_mean = np.nanmean(x)
    if np.ma.is_masked(x):
        # default fill_value in Numpy MaskedArray is 1e20.
        # mean gets masked if entire rows are unknown
        row_mean = np.ma.masked_invalid(row_mean)
        row_mean = np.ma.filled(row_mean, mat_mean)
        indices = np.logical_or(~np.isfinite(x.data), x.mask)
    else:
        row_mean[np.isnan(row_mean)] = mat_mean
        indices = ~np.isfinite(x)
    filled = x.copy()
    filled[indices] = np.take(row_mean, indices.nonzero()[0])
    return filled


def fill_col(x):
    return fill_row(x.T).T


def fill_const(x, const):
    filled = x.copy()
    filled[~np.isfinite(x)] = const
    if np.ma.is_masked(x):
        filled.data[x.mask] = const
    return filled


FILL_CONST = 'const'
FILL_TYPE = dict([
    ('mean', fill_mean),
    ('row_mean', fill_row),
    ('col_mean', fill_col),
    ('const', fill_const)
])


class Relation(object):
    """Relation used for data fusion.

    Attributes
    ----------
    data :
    row_type :
    col_type :
    name :
    row_names :
    col_names :
    fill_value : 'mean', 'row_mean', 'col_mean' or float
    row_metadata :
    col_metadata :
    preprocessor :
    postprocessor :
    """
    def __init__(self, data, row_type, col_type, name='',
                 row_names=None, col_names=None, fill_value='mean',
                 row_metadata=None, col_metadata=None,
                 preprocessor=None, postprocessor=None, **kwargs):
        self.__dict__.update(locals())
        self.__dict__.update(kwargs)
        self.__dict__.pop('kwargs', None)
        self.__dict__.pop('self', None)
        self._id = name or uuid()

    def filled(self):
        if isinstance(self.fill_value, Number):
            filled_data = FILL_TYPE[FILL_CONST](self.data, self.fill_value)
        else:
            filled_data = FILL_TYPE[self.fill_value](self.data)
        return filled_data

    def __contains__(self, obj_type):
        return obj_type == self.row_type or obj_type == self.col_type

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._id == other._id

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.__repr__(str)

    def __repr__(self, repr=repr):
        return "{}({} {} {})".format(
            self.__class__.__name__,
            repr(self.row_type),
            ('"%s"' % self.name) if self.name else "→",
            repr(self.col_type))

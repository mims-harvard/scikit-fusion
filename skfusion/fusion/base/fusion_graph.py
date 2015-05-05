from collections import defaultdict, Iterable

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
        self.relations = {}
        self.object_types = {}
        self.add_relations_from(relations)

    @property
    def n_relations(self):
        return len(self.relations)

    @property
    def n_object_types(self):
        return len(self.object_types)

    def __getitem__(self, key):
        return self.adjacency_matrix[key]

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

    def draw_graphviz(self, filename):
        """Draw the data fusion graph and save it to a file (SVG).

        Parameters
        ----------
        filename :
        """
        import pygraphviz as pgv
        fus_graph = pgv.AGraph(strict=False, directed=True)
        # object types
        for ot in self.object_types:
            fus_graph.add_node(ot.name)
        # relations
        ot2count = defaultdict(int)
        for relation in self.relations:
            ot1 = relation.row_type
            ot2 = relation.col_type
            ot2count[ot1, ot2] += 1
            if ot1 != ot2:
                label = '<<b>R</b><SUB>%s,%s</SUB><SUP>%d</SUP>' \
                        '<br/>>' % (ot1.name, ot2.name, ot2count[ot1, ot2])
                fus_graph.add_edge(ot1.name, ot2.name, text=label)
            else:
                label = '<<b>&Theta;</b><SUB>%s</SUB>' \
                        '<SUP>%d</SUP><br/>>' % (ot1.name, ot2count[ot1, ot2])
                fus_graph.add_edge(ot1.name, ot1.name, label=label)
        fus_graph.draw(filename, format='pdf', prog='dot')

    def add_relation(self, relation):
        """Add a single relation to the fusion graph.

        Parameters
        ----------
        relation :
        """
        self.relations[relation] = relation
        self.object_types[relation.row_type] = relation.row_type
        self.object_types[relation.col_type] = relation.col_type
        neighbors = self.adjacency_matrix.get(relation.row_type, {})
        nbs_list = neighbors.get(relation.col_type, []) + [relation]
        neighbors[relation.col_type] = nbs_list
        self.adjacency_matrix[relation.row_type] = neighbors

    def add_relations_from(self, relations):
        """Add relations to the fusion graph.

        Parameters
        ----------
        relations : container of relations
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
        del self.relations[relation]
        if self.adjacency_matrix[relation.row_type][relation.col_type] == []:
            del self.adjacency_matrix[relation.row_type][relation.col_type]
        if not list(self.in_neighbors(relation.row_type)) and \
                not list(self.out_neighbors(relation.row_type)):
            self.remove_object_type(relation.row_type)
        if not list(self.in_neighbors(relation.col_type)) and \
                not list(self.out_neighbors(relation.col_type)):
            self.remove_object_type(relation.col_type)

    def remove_relations_from(self, relations):
        """Remove relations from the fusion graph.

        Parameters
        ----------
        relations : container of relations
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
            if object_type == relation.row_type or object_type == relation.col_type:
                self.remove_relation(relation)
        del self.adjacency_matrix[object_type]
        for obj_type in self.adjacency_matrix:
            if object_type in self.adjacency_matrix[obj_type]:
                del self.adjacency_matrix[obj_type][object_type]
        del self.object_types[object_type]

    def remove_object_types_from(self, object_types):
        """Remove relations from the fusion graph.

        Parameters
        ----------
        object_types: container of object_types
        """
        for object_type in object_types:
            self.remove_object_type(object_type)

    def get_relations(self, row_type, col_type=None):
        """Return an iterator for relation matrices between two types of objects.

        Parameters
        ----------
        row_type : Object type identifier
        col_type : Object type identifies

        Returns
        -------
        relation :  an iterator
        """
        if row_type not in self.object_types:
            raise DataFusionError("Object types are not recognized.")
        if col_type is not None and col_type not in self.object_types:
            raise DataFusionError("Object types are not recognized.")
        return iter(self.adjacency_matrix.get(row_type, {}).get(col_type, []))

    def get_object_type(self, name):
        """Return object type whose name is provided.

        Parameters
        ----------
        name :
        """
        object_types = [obj_type for obj_type in self.object_types
                       if obj_type.name == name]
        return object_types if len(object_types) != 1 else object_types[0]

    def out_relations(self, object_type):
        """Return an iterator for relations adjacent to the object type.

        Parameters
        ----------
        object_type : Object type identifier

        Returns
        -------
        relation : an iterator
        """
        if object_type not in self.object_types:
            raise DataFusionError("Object type not in the fusion graph.")
        for col_type in self.adjacency_matrix[object_type]:
            for relation in self.adjacency_matrix[object_type][col_type]:
                yield relation

    def in_relations(self, object_type):
        """Return an iterator for relations adjacent to the object type.

        Parameters
        ----------
        object_type : Object type identifier

        Returns
        -------
        relation : an iterator
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
        object_type : Object type identifier

        Returns
        -------
        relation : an iterator
        """
        if object_type not in self.object_types:
            raise DataFusionError("Object type not in the fusion graph.")
        return iter(self.adjacency_matrix.get(object_type, {}).keys())

    def in_neighbors(self, object_type):
        """Return an iterator for object types adjacent to the object type.

        Parameters
        ----------
        object_type : Object type identifier

        Returns
        -------
        relation : an iterator
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
            self.__class__.__name__, repr(list(self.object_types.values())),
            repr(list(self.relations.values())))


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
        return self.name == other

    def __ne__(self, other):
        return self.name != other

    def __str__(self):
        return "{}(\"{}\")".format(self.__class__.__name__, self.name)

    def __repr__(self):
        return "{}(\"{}\")".format(self.__class__.__name__, self.name)


class Relation(object):
    """Relation used for data fusion.

    Attributes
    ----------
    data :
    row_type :
    col_type :
    mask :
    row_names :
    col_names :
    """
    def __init__(self, data, row_type, col_type, mask=None,
                 row_names=None, col_names=None, **kwargs):
        self.__dict__.update(vars())
        self.__dict__.update(kwargs)
        del self.__dict__['kwargs']
        del self.__dict__['self']

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, str(self.row_type), str(self.col_type))

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, repr(self.row_type), repr(self.col_type))

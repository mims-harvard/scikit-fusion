"""
Base code for handling data sets.
"""
import gzip
from os.path import dirname
from os.path import join

import numpy as np

from skfusion.fusion import ObjectType, Relation, FusionGraph


__all__ = ['load_dicty', 'load_pharma']


def load_source(source_path, delimiter=',', filling_value='0'):
    """Load and return a data source.

    Parameters
    ----------
    delimiter : str, optional (default=',')
        The string used to separate values. By default, comma acts as delimiter.

    filling_value : variable, optional (default='0')
        The value to be used as default when the data are missing.

    Returns
    -------
    data : DataSource
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'obj1_names', the meaning of row objects,
        'obj2_names', the meaning of column objects.
    """
    module_path = dirname(__file__)
    data_file = gzip.open(join(module_path, 'data', source_path))
    row_names = np.array(next(data_file).decode('utf-8').strip().split(delimiter))
    col_names = np.array(next(data_file).decode('utf-8').strip().split(delimiter))
    data = np.genfromtxt(data_file, delimiter=delimiter, missing_values=[''],
                         filling_values=filling_value)
    return Relation(data=data, row_names=row_names, col_names=col_names)


def load_dicty():
    """Construct fusion graph from molecular biology of Dictyostelium."""
    ann = load_source(join('dicty', 'dicty.gene_annnotations.csv.gz'))
    expr = load_source(join('dicty', 'dicty.gene_expression.csv.gz'))
    ppi = load_source(join('dicty', 'dicty.ppi.csv.gz'))
    expr.data = np.log(np.maximum(expr.data, np.finfo(np.float).eps))

    return FusionGraph(ann=ann, expr=expr, ppi=ppi,
                        gene=ObjectType('Gene', 50),
                        go_term=ObjectType('GO term', 15),
                        exprc=ObjectType('Exp condition', 5))


def load_pharma():
    """Construct fusion graph from pharmacology domain."""
    actions = load_source(join('pharma', 'pharma.actions.csv.gz'))
    pubmed = load_source(join('pharma', 'pharma.pubmed.csv.gz'))
    depositors = load_source(join('pharma', 'pharma.depositors.csv.gz'))
    fingerprints = load_source(join('pharma', 'pharma.fingerprints.csv.gz'))
    depo_cats = load_source(join('pharma', 'pharma.depo_cats.csv.gz'))
    tanimoto = load_source(join('pharma', 'pharma.tanimoto.csv.gz'))

    return FusionGraph(actions=actions, pubmed=pubmed, depositors=depositors,
                        fingerprints=fingerprints, depo_cats=depo_cats,
                        tanimoto=tanimoto,
                        action=ObjectType('Action', 5),
                        pmid=ObjectType('PMID', 5),
                        depositor=ObjectType('Depositor', 5),
                        fingerprint=ObjectType('Fingerprint', 20),
                        depo_cat=ObjectType('Depositor category', 5),
                        chemical=ObjectType('Chemical', 10))

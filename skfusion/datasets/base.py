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
    return data, row_names, col_names


def load_dicty():
    """Construct fusion graph from molecular biology of Dictyostelium."""
    gene = ObjectType('Gene', 50)
    go_term = ObjectType('GO term', 15)
    exprc = ObjectType('Experimental condition', 5)

    data, rn, cn = load_source(join('dicty', 'dicty.gene_annnotations.csv.gz'))
    ann = Relation(data=data, row_type=gene, col_type=go_term, row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('dicty', 'dicty.gene_expression.csv.gz'))
    expr = Relation(data=data, row_type=gene, col_type=exprc, row_names=rn, col_names=cn)
    expr.data = np.log(np.maximum(expr.data, np.finfo(np.float).eps))
    data, rn, cn = load_source(join('dicty', 'dicty.ppi.csv.gz'))
    ppi = Relation(data=data, row_type=gene, col_type=gene, row_names=rn, col_names=cn)

    return FusionGraph([ann, expr, ppi])


def load_pharma():
    """Construct fusion graph from the pharmacology domain."""
    action=ObjectType('Action', 5)
    pmid=ObjectType('PMID', 5)
    depositor=ObjectType('Depositor', 5)
    fingerprint=ObjectType('Fingerprint', 20)
    depo_cat=ObjectType('Depositor category', 5)
    chemical=ObjectType('Chemical', 10)

    data, rn, cn = load_source(join('pharma', 'pharma.actions.csv.gz'))
    actions = Relation(data=data, row_type=chemical, col_type=action,
                       row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('pharma', 'pharma.pubmed.csv.gz'))
    pubmed = Relation(data=data, row_type=chemical, col_type=pmid,
                      row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('pharma', 'pharma.depositors.csv.gz'))
    depositors = Relation(data=data, row_type=chemical, col_type=depositor,
                          row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('pharma', 'pharma.fingerprints.csv.gz'))
    fingerprints = Relation(data=data, row_type=chemical, col_type=fingerprint,
                            row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('pharma', 'pharma.depo_cats.csv.gz'))
    depo_cats = Relation(data=data, row_type=depositor, col_type=depo_cat,
                         row_names=rn, col_names=cn)
    data, rn, cn = load_source(join('pharma', 'pharma.tanimoto.csv.gz'))
    tanimoto = Relation(data=data, row_type=chemical, col_type=chemical,
                        row_names=rn, col_names=cn)

    return FusionGraph([actions, pubmed, depositors, fingerprints, depo_cats, tanimoto])

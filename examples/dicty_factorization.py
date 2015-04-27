"""
==========================================================================
Fusion of three data sources for gene function prediction in Dictyostelium
==========================================================================

Fuse three data sets: gene expression data (Miranda et al., 2013, PLoS One),
slim gene annotations from Gene Ontology and protein-protein interaction
network from STRING database.

Learnt latent matrix factors are utilized for the prediction of slim GO
terms in Dictyostelium genes that are unavailable in the training phase.

This example demonstrates how predictive performance can
be improved if prediction model relies on features extracted by data fusion
instead of on raw data.
"""
print(__doc__)

from functools import reduce

from sklearn import cross_validation, ensemble, metrics
import numpy as np

from skfusion.datasets import load_dicty
from skfusion import fusion


def mf(dicty, train_idx, test_idx, term_idx):
    max_iter = 10
    n_run = 1
    init_type = "random_vcol"
    random_state = 0

    ann = dicty.ann.data.copy()
    ann[test_idx, :] = 0
    R_train = {(dicty.gene, dicty.go_term): ann,
               (dicty.gene, dicty.exprc): dicty.expr.data}
    T_train = {dicty.gene: [dicty.ppi.data]}

    fuser = fusion.Dfmf()

    p = 0.7
    dicty.gene.rank = p * dicty.ann.data.shape[0]
    dicty.exprc.rank = p * dicty.expr.data.shape[1]
    dicty.go_term.rank = p * dicty.ann.data.shape[1]
    fuser.set_scheme(R_train, T_train)
    fuser.fuse(max_iter, init_type, n_run, random_state=random_state)
    X = fuser.reconstruct(dicty.gene, dicty.exprc)

    X_train = X[train_idx, :]
    y_train = dicty.ann.data[train_idx, term_idx]
    clf = ensemble.RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    X_new = X[test_idx, :]
    y_pred = clf.predict_proba(X_new)[:, 1]
    return y_pred


def rf(dicty, train_idx, test_idx, term_idx):
    X_train = dicty.expr.data[train_idx, :]
    y_train = dicty.ann.data[train_idx, term_idx]
    clf = ensemble.RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    X_new = dicty.expr.data[test_idx, :]
    y_pred = clf.predict_proba(X_new)[:, 1]
    return y_pred


def main():
    n_folds = 10
    dicty = load_dicty()
    n_genes, n_terms = dicty.ann.data.shape

    for t, term_idx in enumerate(range(n_terms)):
        term = dicty.ann.obj2_names[term_idx]
        print("Term: %s" % term)
        y_true = dicty.ann.data[:, term_idx]

        cls_size = int(y_true.sum())
        if cls_size > n_genes - 20 or cls_size < 20:
            continue

        skf = cross_validation.StratifiedKFold(y_true, n_folds=n_folds)
        y_pred_mf = np.zeros_like(y_true)
        y_pred_rf = np.zeros_like(y_true)
        for i, (train_idx, test_idx) in enumerate(skf):
            print("\tFold: %d" % (i+1))
            # Let"s make predictions from fused data representation
            y_pred_mf[test_idx] = mf(dicty, train_idx, test_idx, term_idx)
            # Let"s make predictions from raw data
            y_pred_rf[test_idx] = rf(dicty, train_idx, test_idx, term_idx)

        mfa = metrics.roc_auc_score(y_true, y_pred_mf)
        rfa = metrics.roc_auc_score(y_true, y_pred_rf)
        print("(%2d/%2d): %10s MF: %0.3f RF: %0.3f" % (t+1, n_terms, term, mfa, rfa))


if __name__ == "__main__":
    main()

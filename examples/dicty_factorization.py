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

from sklearn import cross_validation, ensemble, metrics
import numpy as np

from skfusion.datasets import load_dicty
from skfusion import fusion as skf


def mf(train_idx, test_idx, term_idx):
    ann = dicty[gene][go_term][0].data.copy()
    ann[test_idx, :] = 0
    relations = [
        skf.Relation(ann, gene, go_term),
        skf.Relation(dicty[gene][exp_cond][0].data, gene, exp_cond),
        skf.Relation(dicty[gene][gene][0].data, gene, gene)]
    fusion_graph = skf.FusionGraph(relations)

    fuser = skf.Dfmf(max_iter=10, n_run=1, init_type="random_vcol", random_state=0)

    p = 0.7
    gene.rank = p * dicty[gene][go_term][0].data.shape[0]
    exp_cond.rank = p * dicty[gene][exp_cond][0].data.shape[1]
    go_term.rank = p * dicty[gene][go_term][0].data.shape[1]
    fuser.fuse(fusion_graph)
    X = fuser.complete(fusion_graph[gene][exp_cond][0])

    X_train = X[train_idx, :]
    y_train = dicty[gene][go_term][0].data[train_idx, term_idx]
    clf = ensemble.RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    X_new = X[test_idx, :]
    y_pred = clf.predict_proba(X_new)[:, 1]
    return y_pred


def rf(train_idx, test_idx, term_idx):
    X_train = dicty[gene][exp_cond][0].data[train_idx, :]
    y_train = dicty[gene][go_term][0].data[train_idx, term_idx]
    clf = ensemble.RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    X_new = dicty[gene][exp_cond][0].data[test_idx, :]
    y_pred = clf.predict_proba(X_new)[:, 1]
    return y_pred


def main():
    n_folds = 10
    n_genes, n_terms = dicty[gene][go_term][0].data.shape

    for t, term_idx in enumerate(range(n_terms)):
        term = dicty[gene][go_term][0].col_names[term_idx]
        print("Term: %s" % term)
        y_true = dicty[gene][go_term][0].data[:, term_idx]

        cls_size = int(y_true.sum())
        if cls_size > n_genes - 20 or cls_size < 20:
            continue

        skf = cross_validation.StratifiedKFold(y_true, n_folds=n_folds)
        y_pred_mf = np.zeros_like(y_true)
        y_pred_rf = np.zeros_like(y_true)
        for i, (train_idx, test_idx) in enumerate(skf):
            print("\tFold: %d" % (i+1))
            # Let"s make predictions from fused data representation
            y_pred_mf[test_idx] = mf(train_idx, test_idx, term_idx)
            # Let"s make predictions from raw data
            y_pred_rf[test_idx] = rf(train_idx, test_idx, term_idx)

        mfa = metrics.roc_auc_score(y_true, y_pred_mf)
        rfa = metrics.roc_auc_score(y_true, y_pred_rf)
        print("(%2d/%2d): %10s MF: %0.3f RF: %0.3f" % (t+1, n_terms, term, mfa, rfa))


if __name__ == "__main__":
    dicty = load_dicty()
    gene = dicty.get_object_type("Gene")
    go_term = dicty.get_object_type("GO term")
    exp_cond = dicty.get_object_type("Experimental condition")

    main()

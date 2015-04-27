"""
==========================================================================
Fusion of three data sources for gene function prediction in Dictyostelium
==========================================================================

Fuse three data sets: gene expression data (Miranda et al., 2013, PLoS One),
slim gene annotations from Gene Ontology and protein-protein interaction
network from STRING database.

Learnt latent matrix factors are utilized for the prediction of slim GO
terms in Dictyostelium genes that are unavailable in the training phase.

This example demonstrates how chaining of fused latent matrices
along the paths of the fusion graph can be used together with an
established prediction model.
"""
print(__doc__)

from functools import reduce

from sklearn import cross_validation, ensemble, metrics
import numpy as np

from skfusion.datasets import load_dicty
from skfusion import fusion


def fuse(dicty, train_idx):
    max_iter = 300
    n_run = 3
    init_type = "random_vcol"

    R_train = {(dicty.gene, dicty.go_term): dicty.ann.data[train_idx, :],
               (dicty.gene, dicty.exprc): dicty.expr.data[train_idx, :]}
    T_train = {dicty.gene: [dicty.ppi.data[train_idx, :][:, train_idx]]}
    fuser = fusion.Dfmf()
    fuser.set_scheme(R_train, T_train)
    fuser.fuse(max_iter, init_type, n_run)
    return fuser


def profile(fuser, transformer, dicty):
    X = []
    for obj_type in dicty.obj_types():
        for c in transformer.chain(dicty.gene, obj_type):
            X_run = []
            for run in range(fuser.n_run):
                cf = [fuser.backbone(c[i], c[i+1], run) for i in range(len(c)-1)]
                bb = reduce(np.dot, cf) if cf != [] else []
                gene_factor = transformer.factor(dicty.gene, run)
                chained_profile = np.dot(gene_factor, bb) if bb != [] else gene_factor
                X_run.append(chained_profile)
            X.append(reduce(np.add, X_run) / fuser.n_run)
    X = np.hstack(X)
    return X


def transform(fuser, dicty, test_idx):
    R_new = {(dicty.gene, dicty.exprc): dicty.expr.data[test_idx, :]}
    T_new = {dicty.gene: [dicty.ppi.data[test_idx, :][:, test_idx]]}

    transformer = fusion.DfmfTransform(dicty.gene, fuser)
    transformer.set_scheme(R_new, T_new)
    transformer.transform()
    return transformer


def predict_term(dicty, train_idx, test_idx, term_idx):
    fuser = fuse(dicty, train_idx)
    X_train = profile(fuser, fuser, dicty)
    y_train = dicty.ann.data[train_idx, term_idx]
    clf = ensemble.RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)

    transformer = transform(fuser, dicty, test_idx)
    X_test = profile(fuser, transformer, dicty)
    y_pred = clf.predict_proba(X_test)[:, 1]
    return y_pred


def main():
    n_folds = 10
    dicty = load_dicty()
    n_genes, n_terms = dicty.ann.data.shape
    for t, term_idx in enumerate(range(n_terms)):
        y_true = dicty.ann.data[:, term_idx]

        cls_size = int(y_true.sum())
        if cls_size > n_genes - 10 or cls_size < 10:
            continue

        skf = cross_validation.StratifiedKFold(y_true, n_folds=n_folds)
        y_pred = np.zeros_like(y_true)
        for i, (train_idx, test_idx) in enumerate(skf):
            y_pred[test_idx] = predict_term(dicty, train_idx, test_idx, term_idx)

        term_auc = metrics.roc_auc_score(y_true, y_pred)
        term = dicty.ann.obj2_names[term_idx]
        print("(%2d/%2d): %10s AUC: %5.4f" % (t+1, n_terms, term, term_auc))


if __name__ == "__main__":
    main()

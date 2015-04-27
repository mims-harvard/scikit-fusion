"""
==========================================================================
Fusion of three data sources for gene function prediction in Dictyostelium
==========================================================================

Fuse three data sets: gene expression data (Miranda et al., 2013, PLoS One),
slim gene annotations from Gene Ontology and protein-protein interaction
network from STRING database.

Learnt latent matrix factors are utilized for the prediction of slim GO
terms in Dictyostelium genes that are unavailable in the training phase.

This example demonstrates how latent matrices estimated by data fusion
can be utilized for association prediction.
"""
print(__doc__)

from sklearn import cross_validation, metrics
import numpy as np

from mffusion.datasets import load_dicty
from mffusion import dfmf


def predict(X, X_known, obj_idx):
    assoc = np.mean(X[X_known == 1]) if np.sum([X_known == 1]) else 1.
    score = max(0, np.nan_to_num(X[obj_idx]/assoc))
    pred = min(1., score)
    return pred


# Data fusion settings
max_iter = 30
n_run = 3
init_type = 'random'
random_state = 0

# Load data
dicty = load_dicty()
n_genes, n_terms = dicty.ann.data.shape
auc = []

for t, term_idx in enumerate(xrange(n_terms)):
    y = dicty.ann.data[:, term_idx]
    cls_size = int(y.sum())
    if cls_size > n_genes - 20 or cls_size < 20:
        continue
    n_folds = 10

    skf = cross_validation.StratifiedKFold(y, n_folds=n_folds, random_state=random_state)
    term_auc = np.zeros(n_folds)

    for i, (train_idx, test_idx) in enumerate(skf):
        # Specify data fusion scheme for training
        ann = dicty.ann.data.copy()
        ann[test_idx, :] = 0
        R_train = {(dicty.gene, dicty.go_term): ann,
                   (dicty.gene, dicty.exprc): dicty.expr.data}
        T_train = {dicty.gene: [dicty.ppi.data]}

        # Learn the fused latent space
        fuser = dfmf.Dfmf()
        fuser.set_scheme(R_train, T_train)
        fuser.fuse(max_iter, init_type, n_run, random_state=random_state)
        X_new_ann = np.dot(fuser.factor(dicty.gene),
                           np.dot(fuser.backbone(dicty.gene, dicty.go_term),
                                  fuser.factor(dicty.go_term).T))
        # Apply row-centric rule
        pred = [predict(X_new_ann[ti, :], dicty.ann.data[ti, :], term_idx)
                for ti in test_idx]
        term_auc[i] = metrics.roc_auc_score(y[test_idx], pred)

        # from sklearn import ensemble
        # clf1 = ensemble.RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
        #                                       min_samples_split=5, random_state=random_state)
        # clf1.fit(dicty.expr.data[train_idx, :], y[train_idx])
        # X_new = dicty.expr.data[test_idx, :]
        # term_auc[i] = metrics.roc_auc_score(y[test_idx], clf1.predict_proba(X_new)[:, 1])

    # print cross-validated AUC score achieved for GO term
    mfa = np.mean(term_auc)
    auc.append(mfa)
    term = dicty.ann.obj2_names[term_idx]
    print 'Term (%2d/%2d): %10s AUC: %0.4f' % (t+1, n_terms, term, mfa)
print 'Mean AUC across all terms: %0.3f' % np.mean(auc)

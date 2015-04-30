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
from skfusion import fusion as skf


def fuse(train_idx):
    relations = [
        skf.Relation(dicty[gene][go_term][0].data[train_idx, :], gene, go_term),
        skf.Relation(dicty[gene][exp_cond][0].data[train_idx, :], gene, exp_cond),
        skf.Relation(dicty[gene][gene][0].data[train_idx, :][:, train_idx], gene, gene)]
    fusion_graph = skf.FusionGraph(relations)

    fuser = skf.Dfmf(max_iter=50, init_type="random_vcol")
    fuser.fuse(fusion_graph)
    return fuser, fusion_graph


def profile(fuser, transformer):
    X = []
    for obj_type in dicty.object_types:
        for c in fuser.chain(gene, obj_type):
            if obj_type == go_term: continue
            cf = [fuser.backbone(fuser.fusion_graph[c[i]][c[i+1]][0]) for i in range(len(c)-1)]
            bb = reduce(np.dot, cf) if cf != [] else []
            gene_factor = transformer.factor(gene)
            obj_factor = fuser.factor(obj_type)
            chained_profile = np.dot(gene_factor, np.dot(bb, obj_factor.T)) \
                if bb != [] else gene_factor
            X.append(chained_profile)
    X = np.hstack(X)
    return X


def transform(fuser, test_idx):
    relations = [
        skf.Relation(dicty[gene][exp_cond][0].data[test_idx, :], gene, exp_cond),
        skf.Relation(dicty[gene][gene][0].data[test_idx, :][:, test_idx], gene, gene)]
    fusion_graph = skf.FusionGraph(relations)
    transformer = skf.DfmfTransform(max_iter=50, init_type="random_vcol")
    transformer.transform(gene, fusion_graph, fuser)
    return transformer


def predict_term(train_idx, test_idx, term_idx):
    fuser, fuser_graph = fuse(train_idx)
    X_train = profile(fuser, fuser)
    y_train = dicty[gene][go_term][0].data[train_idx, term_idx]
    clf = ensemble.RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    transformer = transform(fuser, test_idx)
    X_test = profile(fuser, transformer)
    y_pred = clf.predict_proba(X_test)[:, 1]
    return y_pred


def main():
    n_folds = 10
    n_genes, n_terms = dicty[gene][go_term][0].data.shape
    for t, term_idx in enumerate(range(n_terms)):
        y_true = dicty[gene][go_term][0].data[:, term_idx]
        cls_size = int(y_true.sum())
        if cls_size > n_genes - 20 or cls_size < 20:
            continue

        skf = cross_validation.StratifiedKFold(y_true, n_folds=n_folds)
        y_pred = np.zeros_like(y_true)
        for i, (train_idx, test_idx) in enumerate(skf):
            y_pred[test_idx] = predict_term(train_idx, test_idx, term_idx)

        term_auc = metrics.roc_auc_score(y_true, y_pred)
        term = dicty[gene][go_term][0].col_names[term_idx]
        print("(%2d/%2d): %10s AUC: %5.4f" % (t+1, n_terms, term, term_auc))


if __name__ == "__main__":
    dicty = load_dicty()
    gene = dicty.get_object_type("Gene")
    go_term = dicty.get_object_type("GO term")
    exp_cond = dicty.get_object_type("Experimental condition")

    main()

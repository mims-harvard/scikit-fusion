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

from skfusion import datasets
from skfusion import fusion as skf


def main():
    n_folds = 10
    n_genes = dicty[gene][go_term][0].data.shape[0]
    cv = cross_validation.KFold(n_genes, n_folds=n_folds)
    fold_mse = np.zeros(n_folds)
    ann_mask = np.zeros_like(dicty[gene][go_term][0].data).astype('bool')

    relations = [
        skf.Relation(dicty[gene][go_term][0].data, gene, go_term),
        skf.Relation(dicty[gene][exp_cond][0].data, gene, exp_cond),
        skf.Relation(dicty[gene][gene][0].data, gene, gene)]
    fusion_graph = skf.FusionGraph(relations)
    fuser = skf.Dfmc(max_iter=30, n_run=1, init_type='random', random_state=0)

    for i, (train_idx, test_idx) in enumerate(cv):
        ann_mask[:] = False
        ann_mask[test_idx, :] = True
        fusion_graph[gene][go_term][0].mask = ann_mask

        fuser.fuse(fusion_graph)
        pred_ann = fuser.complete(fuser.fusion_graph[gene][go_term][0])[test_idx]
        true_ann = dicty[gene][go_term][0].data[test_idx]
        fold_mse[i] = metrics.mean_squared_error(pred_ann, true_ann)

    print("MSE: %5.4f" % np.mean(fold_mse))


if __name__ == "__main__":
    dicty = datasets.load_dicty()
    gene = dicty.get_object_type("Gene")
    go_term = dicty.get_object_type("GO term")
    exp_cond = dicty.get_object_type("Experimental condition")

    main()

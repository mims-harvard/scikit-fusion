"""
================================================================
Fusion of six data sources for pharmacological action prediction
================================================================

This example demonstrates how latent matrices estimated by data
fusion can be utilized for association prediction. The prediction
is done by random forest on features construction through reverse
chaining.
"""
print(__doc__)

from functools import reduce

from sklearn import cross_validation, metrics, ensemble
import numpy as np

from skfusion import datasets
from skfusion import fusion as skf


def fuse(train_idx):
    action_data = pharma[chemical][action][0].data[train_idx]
    pubmed_data = pharma[chemical][pmid][0].data[train_idx]
    depositor_data = pharma[chemical][depositor][0].data[train_idx]
    fingerprint_data = pharma[chemical][fingerprint][0].data[train_idx]
    depo_cat_data = pharma[depositor][depo_cat][0].data
    chemical_data = pharma[chemical][chemical][0].data[train_idx, :][:, train_idx]
    relations = [
        skf.Relation(action_data, chemical, action),
        skf.Relation(pubmed_data, chemical, pmid),
        skf.Relation(depositor_data, chemical, depositor),
        skf.Relation(fingerprint_data, chemical, fingerprint),
        skf.Relation(depo_cat_data, depositor, depo_cat),
        skf.Relation(chemical_data, chemical, chemical)]
    fusion_graph = skf.FusionGraph(relations)

    fuser = skf.Dfmf(max_iter=200, init_type="random_vcol", random_state=0)
    fuser.fuse(fusion_graph)
    return fuser


def profile(fuser, transformer):
    X = []
    for obj_type in pharma.object_types:
        for c in fuser.chain(chemical, obj_type):
            cf = [fuser.backbone(fuser.fusion_graph[c[i]][c[i+1]][0]) for i in range(len(c)-1)]
            bb = reduce(np.dot, cf) if cf != [] else []
            chem_factor = transformer.factor(chemical)
            chained_profile = np.dot(chem_factor, bb) if bb != [] else chem_factor
            X.append(chained_profile)
    X = np.hstack(X)
    return X


def transform(fuser, test_idx):
    pubmed_data = pharma[chemical][pmid][0].data[test_idx]
    depositor_data = pharma[chemical][depositor][0].data[test_idx]
    fingerprint_data = pharma[chemical][fingerprint][0].data[test_idx]
    chemical_data = pharma[chemical][chemical][0].data[test_idx, :][:, test_idx]
    relations = [
        skf.Relation(pubmed_data, chemical, pmid),
        skf.Relation(depositor_data, chemical, depositor),
        skf.Relation(fingerprint_data, chemical, fingerprint),
        skf.Relation(chemical_data, chemical, chemical)]
    fusion_graph = skf.FusionGraph(relations)

    transformer = skf.DfmfTransform(max_iter=200, init_type="random_vcol", random_state=0)
    transformer.transform(chemical, fusion_graph, fuser)
    return transformer


def predict_action(train_idx, test_idx, action_idx):
    fuser = fuse(train_idx)
    X_train = profile(fuser, fuser)
    y_train = pharma[chemical][action][0].data[train_idx, action_idx]
    clf = ensemble.RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)

    transformer = transform(fuser, test_idx)
    X_test = profile(fuser, transformer)
    y_pred = clf.predict_proba(X_test)[:, 1]
    return y_pred


def main():
    n_folds = 10
    n_chemicals, n_actions = pharma[chemical][action][0].data.shape
    for t, action_idx in enumerate(range(n_actions)):
        y_true = pharma[chemical][action][0].data[:, action_idx]
        cls_size = int(y_true.sum())
        if cls_size > n_chemicals - 20 or cls_size < 20:
            continue

        cv = cross_validation.StratifiedKFold(y_true, n_folds=n_folds)
        y_pred = np.zeros_like(y_true)
        for i, (train_idx, test_idx) in enumerate(cv):
            y_pred[test_idx] = predict_action(train_idx, test_idx, action_idx)

        action_auc = metrics.roc_auc_score(y_true, y_pred)
        action_name = pharma[chemical][action][0].col_names[action_idx]
        print("(%2d/%2d): %-30s AUC: %0.4f" % (t+1, n_actions, action_name, action_auc))


if __name__ == "__main__":
    pharma = datasets.load_pharma()
    action = pharma.get_object_type('Action')
    pmid = pharma.get_object_type('PMID')
    depositor = pharma.get_object_type('Depositor')
    fingerprint = pharma.get_object_type('Fingerprint')
    depo_cat = pharma.get_object_type('Depositor category')
    chemical = pharma.get_object_type('Chemical')

    main()

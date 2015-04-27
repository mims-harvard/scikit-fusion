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

from skfusion.datasets import load_pharma
from skfusion import fusion


def fuse(pharma, train_idx):
    max_iter = 200
    n_run = 1
    init_type = "random_vcol"
    rnd_state = 0

    R_train = {(pharma.chemical, pharma.action): pharma.actions.data[train_idx],
               (pharma.chemical, pharma.pmid) : pharma.pubmed.data[train_idx],
               (pharma.chemical, pharma.depositor): pharma.depositors.data[train_idx],
               (pharma.chemical, pharma.fingerprint): pharma.fingerprints.data[train_idx],
               (pharma.depositor, pharma.depo_cat): pharma.depo_cats.data}
    T_train = {pharma.chemical: [pharma.tanimoto.data[train_idx, :][:, train_idx]]}

    fuser = fusion.Dfmf()
    fuser.set_scheme(R_train, T_train)
    fuser.fuse(max_iter=max_iter, init_type=init_type, n_run=n_run, random_state=rnd_state)
    return fuser


def profile(fuser, transformer, pharma):
    X = []
    for obj_type in pharma.obj_types():
        for chain in transformer.chain(pharma.chemical, obj_type):
            cf = [fuser.backbone(chain[i], chain[i+1]) for i in range(len(chain)-1)]
            bb = reduce(np.dot, cf) if cf != [] else []
            chem_factor = transformer.factor(pharma.chemical)
            chained_profile = np.dot(chem_factor, bb) if bb != [] else chem_factor
            X.append(chained_profile)
    X = np.hstack(X)
    return X


def transform(fuser, pharma, test_idx):
    R_new = {(pharma.chemical, pharma.pmid) : pharma.pubmed.data[test_idx],
             (pharma.chemical, pharma.depositor): pharma.depositors.data[test_idx],
             (pharma.chemical, pharma.fingerprint): pharma.fingerprints.data[test_idx]}
    T_new = {pharma.chemical: [pharma.tanimoto.data[test_idx, :][:, test_idx]]}

    transformer = fusion.DfmfTransform(pharma.chemical, fuser)
    transformer.set_scheme(R_new, T_new)
    transformer.transform()
    return transformer


def predict_action(pharma, train_idx, test_idx, action_idx):
    fuser = fuse(pharma, train_idx)
    X_train = profile(fuser, fuser, pharma)
    y_train = pharma.actions.data[train_idx, action_idx]
    clf = ensemble.RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)

    transformer = transform(fuser, pharma, test_idx)
    X_test = profile(fuser, transformer, pharma)
    y_pred = clf.predict_proba(X_test)[:, 1]
    return y_pred


def main():
    n_folds = 10
    pharma = load_pharma()
    n_chemicals, n_actions = pharma.actions.data.shape
    for t, action_idx in enumerate(range(n_actions)):
        y_true = pharma.actions.data[:, action_idx]

        cls_size = int(y_true.sum())
        if cls_size > n_chemicals - 20 or cls_size < 20:
            continue

        skf = cross_validation.StratifiedKFold(y_true, n_folds=n_folds)
        y_pred = np.zeros_like(y_true)
        for i, (train_idx, test_idx) in enumerate(skf):
            y_pred[test_idx] = predict_action(pharma, train_idx, test_idx, action_idx)

        action_auc = metrics.roc_auc_score(y_true, y_pred)
        action = pharma.actions.obj2_names[action_idx]
        print("(%2d/%2d): %-30s AUC: %0.4f" % (t+1, n_actions, action, action_auc))


if __name__ == "__main__":
    main()

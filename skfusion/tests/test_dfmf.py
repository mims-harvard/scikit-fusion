import unittest

import numpy as np
from skfusion.fusion import Relation, ObjectType, \
    FusionGraph, Dfmf, DfmfTransform


class TestDfmf(unittest.TestCase):
    def test_dfmf(self):
        rnds = np.random.RandomState(0)
        R12 = rnds.rand(50, 30)

        t1 = ObjectType('type1', 50)
        t2 = ObjectType('type2', 30)
        relation = Relation(R12, t1, t2)
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(relation)

        fuser = Dfmf(init_type='random', random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.backbone(relation).shape, (50, 30))
        self.assertEqual(fuser.factor(t1).shape, (50, 50))
        self.assertEqual(fuser.factor(t2).shape, (30, 30))
        np.testing.assert_almost_equal(fuser.complete(relation), relation.data)

    def test_infinite(self):
        rnds = np.random.RandomState(0)
        R12 = rnds.rand(50, 30)
        R13 = rnds.rand(50, 10)
        R12 = np.ma.masked_greater(R12, 0.7)
        R12[R12 < 0.1] = np.nan
        R13[R13 < 0.5] = np.inf

        t1 = ObjectType('type1', 50)
        t2 = ObjectType('type2', 30)
        t3 = ObjectType('type3', 10)
        relations = [Relation(R12, t1, t2, fill_value='row_mean'),
                     Relation(R13, t1, t3, fill_value='col_mean')]
        fusion_graph = FusionGraph(relations)

        fuser = Dfmf(init_type='random', random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.backbone(relations[0]).shape, (50, 30))
        self.assertEqual(fuser.backbone(relations[1]).shape, (50, 10))
        self.assertEqual(fuser.factor(t1).shape, (50, 50))
        self.assertEqual(fuser.factor(t2).shape, (30, 30))
        size = np.sum(np.isfinite(fuser.complete(relations[0])))
        np.testing.assert_equal(size, R12.size)

    def test_transformation(self):
        R12 = np.random.rand(5, 3)

        t1 = ObjectType('type1', 2)
        t2 = ObjectType('type2', 2)
        relation = Relation(R12, t1, t2)
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(relation)

        rnds = np.random.RandomState(0)
        fuser = Dfmf(init_type='random', random_state=rnds, max_iter=100
                     ).fuse(fusion_graph)

        new_R12 = R12[:2].copy()
        new_graph = FusionGraph([Relation(new_R12, t1, t2)])

        new_rnds = np.random.RandomState(0)
        transformer = DfmfTransform(random_state=new_rnds).transform(
            t1, new_graph, fuser)

        new_G1 = transformer.factor(t1)
        G1 = fuser.factor(t1)
        G2 = fuser.factor(t2)
        S12 = fuser.backbone(relation)
        new_R12_hat = np.dot(new_G1, np.dot(S12, G2.T))
        R12_hat = np.dot(G1, np.dot(S12, G2.T))

        diff_G1 = new_G1 - G1[:2]
        diff_hat = new_R12_hat - R12_hat[:2]
        self.assertLess(np.sum(diff_G1 ** 2) / diff_G1.size, 1e-5)
        self.assertLess(np.sum(diff_hat ** 2) / diff_hat.size, 1e-5)

    def test_preprocessors(self):
        rnds = np.random.RandomState(0)
        R12 = rnds.rand(50, 30)

        t1 = ObjectType('type1', 50)
        t2 = ObjectType('type2', 30)

        def preprocessor(data):
            return np.ones_like(data)

        relation = Relation(R12, t1, t2, preprocessor=preprocessor)
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(relation)

        fuser = Dfmf(init_type='random', random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.backbone(relation).shape, (50, 30))
        self.assertEqual(fuser.factor(t1).shape, (50, 50))
        self.assertEqual(fuser.factor(t2).shape, (30, 30))
        trnf = np.ones_like(relation.data)
        np.testing.assert_almost_equal(fuser.complete(relation), trnf)

    def test_postprocessors(self):
        rnds = np.random.RandomState(0)
        R12 = rnds.rand(50, 30)

        t1 = ObjectType('type1', 50)
        t2 = ObjectType('type2', 30)

        def postprocessor(data):
            return data - np.mean(data)

        relation = Relation(R12, t1, t2, postprocessor=postprocessor)
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(relation)

        fuser = Dfmf(init_type='random', random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.backbone(relation).shape, (50, 30))
        self.assertEqual(fuser.factor(t1).shape, (50, 50))
        self.assertEqual(fuser.factor(t2).shape, (30, 30))
        trnf = relation.data - np.mean(relation.data)
        np.testing.assert_almost_equal(fuser.complete(relation), trnf)


if __name__ == "__main__":
    unittest.main()

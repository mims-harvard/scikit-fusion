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
        fusion_graph.add(relation)

        fuser = Dfmf(fusion_graph).fuse(init_type='random', random_state=rnds)
        self.assertEqual(fuser.backbone(relation).shape, (50, 30))
        self.assertEqual(fuser.factor(t1).shape, (50, 50))
        self.assertEqual(fuser.factor(t2).shape, (30, 30))
        np.testing.assert_almost_equal(fuser.complete(relation), relation.data)

    def test_transformation(self):
        R12 = np.random.rand(5, 3)

        t1 = ObjectType('type1', 2)
        t2 = ObjectType('type2', 2)
        relation = Relation(R12, t1, t2)
        fusion_graph = FusionGraph()
        fusion_graph.add(relation)

        rnds = np.random.RandomState(0)
        fuser = Dfmf(fusion_graph).fuse(init_type='random', random_state=rnds,
                                        max_iter=100)

        new_R12 = R12[:2].copy()
        new_graph = FusionGraph(Relation(new_R12, t1, t2))

        new_rnds = np.random.RandomState(0)
        transformer = DfmfTransform(t1, new_graph, fuser).transform(
            random_state=new_rnds)

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


if __name__ == "__main__":
    unittest.main()

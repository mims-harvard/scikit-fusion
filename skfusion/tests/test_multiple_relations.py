import unittest

import numpy as np
from skfusion.fusion import Relation, ObjectType, \
    FusionGraph, Dfmf, DfmfTransform, Dfmc


class TestMultipleRelations(unittest.TestCase):
    def test_dfmf(self):
        rnds = np.random.RandomState(0)
        R12_1 = np.random.rand(30, 30)
        R12_2 = np.random.rand(30, 30)
        R13 = np.random.rand(30, 20)

        t1 = ObjectType('type1', 30)
        t2 = ObjectType('type2', 30)
        t3 = ObjectType('type3', 20)
        relations = [Relation(R12_1, t1, t2), Relation(R12_2, t1, t2),
                     Relation(R13, t1, t3)]
        fusion_graph = FusionGraph()
        fusion_graph.add_relations_from(relations)
        self.assertEqual(len(fusion_graph.relations), 3)
        self.assertEqual(len(fusion_graph.object_types), 3)

        fuser = Dfmf(init_type='random', random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.backbone(relations[0]).shape, (30, 30))
        self.assertEqual(fuser.backbone(relations[1]).shape, (30, 30))
        self.assertEqual(fuser.backbone(relations[2]).shape, (30, 20))
        G1 = fuser.factor(t1)
        G2 = fuser.factor(t2)
        S12_1 = fuser.backbone(relations[0])
        S12_2 = fuser.backbone(relations[1])
        R12_1_hat = np.dot(G1, np.dot(S12_1, G2.T))
        R12_2_hat = np.dot(G1, np.dot(S12_2, G2.T))
        np.testing.assert_almost_equal(fuser.complete(relations[0]), R12_1_hat)
        np.testing.assert_almost_equal(fuser.complete(relations[1]), R12_2_hat)

    def test_dfmc(self):
        rnds = np.random.RandomState(0)
        R12_1 = np.random.rand(30, 30)
        R12_2 = np.random.rand(30, 30)
        R13 = np.random.rand(30, 20)

        t1 = ObjectType('type1', 30)
        t2 = ObjectType('type2', 30)
        t3 = ObjectType('type3', 20)
        relations = [Relation(R12_1, t1, t2), Relation(R12_2, t1, t2),
                     Relation(R13, t1, t3)]
        fusion_graph = FusionGraph()
        fusion_graph.add_relations_from(relations)
        self.assertEqual(len(fusion_graph.relations), 3)
        self.assertEqual(len(fusion_graph.object_types), 3)

        fuser = Dfmc(init_type='random', random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.backbone(relations[0]).shape, (30, 30))
        self.assertEqual(fuser.backbone(relations[1]).shape, (30, 30))
        self.assertEqual(fuser.backbone(relations[2]).shape, (30, 20))
        G1 = fuser.factor(t1)
        G2 = fuser.factor(t2)
        S12_1 = fuser.backbone(relations[0])
        S12_2 = fuser.backbone(relations[1])
        R12_1_hat = np.dot(G1, np.dot(S12_1, G2.T))
        R12_2_hat = np.dot(G1, np.dot(S12_2, G2.T))
        np.testing.assert_almost_equal(fuser.complete(relations[0]), R12_1_hat)
        np.testing.assert_almost_equal(fuser.complete(relations[1]), R12_2_hat)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
from skfusion.fusion import Relation, ObjectType, \
    FusionGraph, Dfmf, Dfmc


class TestNRun(unittest.TestCase):
    def test_dfmf(self):
        rnds = np.random.RandomState(0)
        R12 = rnds.rand(30, 30)
        R13 = rnds.rand(30, 30)

        t1 = ObjectType('type1', 50)
        t2 = ObjectType('type2', 30)
        t3 = ObjectType('type3', 10)
        fusion_graph = FusionGraph()
        relations = [Relation(R12, t1, t2), Relation(R13, t1, t3)]
        fusion_graph.add_relations_from(relations)

        fuser = Dfmf(init_type='random', random_state=rnds, n_run=3
                     ).fuse(fusion_graph)
        self.assertEqual(len(list(fuser.factor(t1))), 3)
        self.assertEqual(len(list(fuser.factor(t2))), 3)
        self.assertEqual(len(list(fuser.factor(t3))), 3)
        self.assertEqual(len(list(fuser.backbone(relations[0]))), 3)
        self.assertEqual(len(list(fuser.backbone(relations[1]))), 3)
        for object_type in [t1, t2, t3]:
            for factor in fuser.factor(object_type):
                self.assertEqual(factor.shape, (30, object_type.rank))

        G1 = fuser.factor(t1, run=1)
        S13 = fuser.backbone(relations[1], run=1)
        G3 = fuser.factor(t3, run=1)
        R13_hat = np.dot(G1, np.dot(S13, G3.T))
        completed = fuser.complete(relations[1], run=1)
        np.testing.assert_almost_equal(completed, R13_hat)


    def test_dfmc(self):
        rnds = np.random.RandomState(0)
        R12 = rnds.rand(30, 30)
        R13 = rnds.rand(30, 30)

        t1 = ObjectType('type1', 50)
        t2 = ObjectType('type2', 30)
        t3 = ObjectType('type3', 10)
        fusion_graph = FusionGraph()
        relations = [Relation(R12, t1, t2), Relation(R13, t1, t3)]
        fusion_graph.add_relations_from(relations)

        fuser = Dfmc(init_type='random', random_state=rnds, n_run=3
                     ).fuse(fusion_graph)
        self.assertEqual(len(list(fuser.factor(t1))), 3)
        self.assertEqual(len(list(fuser.factor(t2))), 3)
        self.assertEqual(len(list(fuser.factor(t3))), 3)
        self.assertEqual(len(list(fuser.backbone(relations[0]))), 3)
        self.assertEqual(len(list(fuser.backbone(relations[1]))), 3)
        for object_type in [t1, t2, t3]:
            for factor in fuser.factor(object_type):
                self.assertEqual(factor.shape, (30, object_type.rank))

        G1 = fuser.factor(t1, run=1)
        S13 = fuser.backbone(relations[1], run=1)
        G3 = fuser.factor(t3, run=1)
        R13_hat = np.dot(G1, np.dot(S13, G3.T))
        completed = fuser.complete(relations[1], run=1)
        np.testing.assert_almost_equal(completed, R13_hat)


if __name__ == "__main__":
    unittest.main()

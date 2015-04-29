import unittest

import numpy as np
from skfusion.fusion import Relation, ObjectType, \
    FusionGraph, Dfmc


class TestDfmc(unittest.TestCase):
    def test_dfmc(self):
        rnds = np.random.RandomState(0)
        R12 = rnds.rand(50, 30)

        t1 = ObjectType('type1', 50)
        t2 = ObjectType('type2', 30)
        relation = Relation(R12, t1, t2)
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(relation)

        fuser = Dfmc(init_type='random', random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.backbone(relation).shape, (50, 30))
        self.assertEqual(fuser.factor(t1).shape, (50, 50))
        self.assertEqual(fuser.factor(t2).shape, (30, 30))
        np.testing.assert_almost_equal(fuser.complete(relation), relation.data)

    def test_masked(self):
        rnds = np.random.RandomState(0)
        R12 = rnds.rand(50, 30)
        mask = R12 < 0.3

        t1 = ObjectType('type1', 50)
        t2 = ObjectType('type2', 30)
        relation = Relation(R12, t1, t2, mask=mask)
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(relation)

        fuser = Dfmc(init_type='random', random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.backbone(relation).shape, (50, 30))
        self.assertEqual(fuser.factor(t1).shape, (50, 50))
        self.assertEqual(fuser.factor(t2).shape, (30, 30))
        np.testing.assert_almost_equal(fuser.complete(relation)[~mask], relation.data[~mask])


if __name__ == "__main__":
    unittest.main()

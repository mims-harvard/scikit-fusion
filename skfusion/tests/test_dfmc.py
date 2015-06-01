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
        R12 = np.ma.masked_less(rnds.rand(50, 30), 0.3)

        t1 = ObjectType('type1', 50)
        t2 = ObjectType('type2', 30)
        relation = Relation(R12, t1, t2)
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(relation)

        fuser = Dfmc(init_type='random', random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.backbone(relation).shape, (50, 30))
        self.assertEqual(fuser.factor(t1).shape, (50, 50))
        self.assertEqual(fuser.factor(t2).shape, (30, 30))
        np.testing.assert_almost_equal(fuser.complete(relation)[~R12.mask], relation.data[~R12.mask])

    def test_preprocessors(self):
        rnds = np.random.RandomState(0)
        R12 = rnds.rand(50, 30)
        R12 = np.ma.masked_greater(R12, 0.7)

        t1 = ObjectType('type1', 50)
        t2 = ObjectType('type2', 30)

        def preprocessor(data):
            return np.ones_like(data)

        relation = Relation(R12, t1, t2, name='R', preprocessor=preprocessor)
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(relation)

        fuser = Dfmc(init_type='random', random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.backbone(relation).shape, (50, 30))
        self.assertEqual(fuser.factor(t1).shape, (50, 50))
        self.assertEqual(fuser.factor(t2).shape, (30, 30))
        trnf = np.ones_like(relation.data)
        np.testing.assert_almost_equal(fuser.complete(relation), trnf)
        np.testing.assert_equal(fusion_graph.get_relation('R').data, R12)

    def test_postprocessors(self):
        rnds = np.random.RandomState(0)
        R12 = rnds.rand(50, 30)
        R12 = np.ma.masked_greater(R12, 0.7)

        t1 = ObjectType('type1', 50)
        t2 = ObjectType('type2', 30)

        def postprocessor(data):
            return data - 10

        relation = Relation(R12, t1, t2, name='R', postprocessor=postprocessor)
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(relation)

        fuser = Dfmc(init_type='random', random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.backbone(relation).shape, (50, 30))
        self.assertEqual(fuser.factor(t1).shape, (50, 50))
        self.assertEqual(fuser.factor(t2).shape, (30, 30))
        trnf = relation.data - 10
        np.testing.assert_almost_equal(fuser.complete(relation), trnf)
        np.testing.assert_equal(fusion_graph.get_relation('R').data, R12)


if __name__ == "__main__":
    unittest.main()

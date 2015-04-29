import unittest

import numpy as np
from skfusion.fusion import Relation, ObjectType, \
    FusionGraph, Dfmf, DfmfTransform


class TestBase(unittest.TestCase):
    def test_pipeline(self):
        rnds = np.random.RandomState(0)
        R12 = rnds.rand(50, 30)
        R13 = rnds.rand(50, 40)
        R23 = rnds.rand(30, 40)

        t1 = ObjectType('type1', 30)
        t2 = ObjectType('type2', 40)
        t3 = ObjectType('type3', 40)
        relations = [Relation(R12, t1, t2),
                     Relation(R13, t1, t3), Relation(R23, t2, t3)]
        fusion_graph = FusionGraph()
        fusion_graph.add_relations_from(relations)

        fuser = Dfmf(random_state=rnds).fuse(fusion_graph)
        self.assertEqual(fuser.factor(t1).shape, (50, 30))
        self.assertEqual(fuser.factor(t2).shape, (30, 40))
        self.assertEqual(fuser.factor(t3).shape, (40, 40))
        self.assertEqual(fuser.backbone(relations[0]).shape, (30, 40))
        self.assertEqual(fuser.backbone(relations[1]).shape, (30, 40))
        self.assertEqual(fuser.backbone(relations[2]).shape, (40, 40))

        new_R12 = rnds.rand(15, 30)
        new_R13 = rnds.rand(15, 40)

        new_relations = [Relation(new_R12, t1, t2), Relation(new_R13, t1, t3)]
        new_graph = FusionGraph(new_relations)

        transformer = DfmfTransform(random_state=rnds).transform(t1, new_graph, fuser)
        self.assertEqual(transformer.factor(t1).shape, (15, 30))


if __name__ == "__main__":
    unittest.main()

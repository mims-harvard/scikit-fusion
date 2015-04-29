import unittest

import numpy as np
from skfusion.fusion import Relation, ObjectType, \
    FusionGraph


class TestFusionGraph(unittest.TestCase):
    def test_dfmf(self):
        R12 = np.random.rand(30, 30)
        R23 = np.random.rand(30, 30)
        R34 = np.random.rand(30, 30)
        R45 = np.random.rand(30, 30)
        R35 = np.random.rand(30, 30)
        R51 = np.random.rand(30, 30)

        t1 = ObjectType('Type 1', 10)
        t2 = ObjectType('Type 2', 10)
        t3 = ObjectType('Type 3', 10)
        t4 = ObjectType('Type 4', 10)
        t5 = ObjectType('Type 5', 10)
        relations = [Relation(R12, t1, t2), Relation(R23, t2, t3),
                     Relation(R34, t3, t4), Relation(R45, t4, t5),
                     Relation(R35, t3, t5), Relation(R51, t5, t1)]
        fusion_graph = FusionGraph()
        fusion_graph.add_relations_from(relations)
        # fusion_graph.draw_graphviz('test_fusion_graph.pdf')


if __name__ == "__main__":
    unittest.main()

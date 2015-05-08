import unittest

import numpy as np
from skfusion.fusion import Relation, ObjectType, \
    FusionGraph


class TestFusionGraph(unittest.TestCase):
    def setUp(self):
        rnds = np.random.RandomState(0)
        X = rnds.rand(30, 30)
        self.t1 = ObjectType('Type 1', 10)
        self.t2 = ObjectType('Type 2', 10)
        self.t3 = ObjectType('Type 3', 10)
        self.t4 = ObjectType('Type 4', 10)
        self.t5 = ObjectType('Type 5', 10)
        self.relations1 = [
            Relation(X, self.t1, self.t2, name='Test1'), Relation(X, self.t2, self.t3),
            Relation(X, self.t3, self.t4), Relation(X, self.t4, self.t5),
            Relation(X, self.t3, self.t5), Relation(X, self.t5, self.t1)]

        self.relations2 = [
            Relation(X, self.t1, self.t2, name='Test2'), Relation(X, self.t1, self.t2),
            Relation(X, self.t2, self.t3), Relation(X, self.t3, self.t4),
            Relation(X, self.t4, self.t5), Relation(X, self.t3, self.t5),
            Relation(X, self.t5, self.t1), Relation(X, self.t4, self.t4),
            Relation(X, self.t4, self.t4, name='Test3'), Relation(X, self.t5, self.t5)]

    def test_drawing(self):
        fusion_graph = FusionGraph()
        fusion_graph.add_relations_from(self.relations1)
        # fusion_graph.draw_graphviz('test_fusion_graph.pdf')

    def test_manipulation(self):
        fusion_graph = FusionGraph()
        fusion_graph.add_relations_from(self.relations2)

        self.assertEqual(fusion_graph['Test2'], self.relations2[0])
        self.assertEqual(fusion_graph['Test3'], self.relations2[8])

        self.assertEqual(fusion_graph.n_object_types, 5)
        self.assertEqual(fusion_graph.n_relations, 10)

        fusion_graph.remove_relation(self.relations2[6])
        self.assertEqual(fusion_graph.n_object_types, 5)
        self.assertEqual(fusion_graph.n_relations, 9)

        fusion_graph.remove_relations_from([
            self.relations2[9], self.relations2[4], self.relations2[5]])
        self.assertEqual(fusion_graph.n_object_types, 4)
        self.assertEqual(fusion_graph.n_relations, 6)

    def test_inspection(self):
        fusion_graph = FusionGraph(self.relations2)
        self.assertEqual(set(fusion_graph.in_relations(self.t1)), {self.relations2[6]})
        self.assertEqual(set(fusion_graph.out_relations(self.t1)), set(self.relations2[:2]))
        out_nbs = {self.relations2[4], self.relations2[7], self.relations2[8]}
        self.assertEqual(set(fusion_graph.out_relations(self.t4)), out_nbs)

    def test_retrieval(self):
        fusion_graph = FusionGraph(self.relations2)
        self.assertEqual(fusion_graph.get_object_type('Type 1'), self.t1)
        self.assertEqual(list(fusion_graph.get_relations(self.t1, self.t2)), self.relations2[:2])
        self.assertEqual(fusion_graph[self.t1][self.t2], self.relations2[:2])
        out_degree1 = len(list(fusion_graph.out_relations(self.t4)))
        out_degree2 = sum(len(rels) for rels in fusion_graph[self.t4].values())
        self.assertEqual(out_degree1, out_degree2)


if __name__ == "__main__":
    unittest.main()

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

    def test_removal_single_relation(self):
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(self.relations1[0])
        self.assertEqual(fusion_graph.n_relations, 1)
        self.assertEqual(fusion_graph.n_object_types, 2)
        fusion_graph.remove_relation(self.relations1[0])
        self.assertEqual(fusion_graph.n_relations, 0)
        self.assertEqual(fusion_graph.n_object_types, 0)

    def test_removal_of_loops(self):
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(self.relations2[-1])
        self.assertEqual(fusion_graph.n_relations, 1)
        self.assertEqual(fusion_graph.n_object_types, 1)
        fusion_graph.remove_relation(self.relations2[-1])
        self.assertEqual(fusion_graph.n_relations, 0)
        self.assertEqual(fusion_graph.n_object_types, 0)

    def test_get_names_by_object_type(self):
        rnds = np.random.RandomState(0)
        X = rnds.rand(10, 10)
        t1_names = list('ABCDEFGHIJ')
        t2_names = list('KLMNOPQRST')

        rel = Relation(X, name='Test',
                       row_type=self.t1, row_names=t1_names,
                       col_type=self.t2, col_names=t2_names)
        rel2 = Relation(X, name='Test2',
                        row_type=self.t2, row_names=t2_names,
                        col_type=self.t3)
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(rel)
        fusion_graph.add_relation(rel2)

        self.assertEqual(fusion_graph.get_names(self.t1), t1_names)
        self.assertEqual(fusion_graph.get_names(self.t2), t2_names)
        t3_names = fusion_graph.get_names(self.t3)
        self.assertEqual(len(t3_names), 10)

    def test_get_object_type_metadata(self):
        rnds = np.random.RandomState(0)
        X = rnds.rand(10, 10)
        a, b, c = list('ABCDEFGHIJ'), list('0123456789'), list('KLMNOPQRST')
        t1_metadata = [{'a': x} for x in a]
        t2_metadata = [{'b': x} for x in b]
        t2_metadata2 = [{'d': x} for x in b]

        rel = Relation(X, name='Test',
                       row_type=self.t1, row_metadata=t1_metadata,
                       col_type=self.t2, col_metadata=t2_metadata)
        rel2 = Relation(X, name='Test2',
                        row_type=self.t2, row_metadata=t2_metadata2,
                        col_type=self.t3)
        fusion_graph = FusionGraph()
        fusion_graph.add_relation(rel)
        fusion_graph.add_relation(rel2)

        def merge(d1, d2):
            d = {}
            d.update(d1)
            d.update(d2)
            return d

        self.assertEqual(fusion_graph.get_metadata(self.t1), t1_metadata)
        self.assertEqual(fusion_graph.get_metadata(self.t2), list(map(merge, t2_metadata, t2_metadata2)))
        t3_metadata = fusion_graph.get_metadata(self.t3)
        self.assertEqual(len(t3_metadata), 10)
        for md in t3_metadata:
            self.assertFalse(md)


if __name__ == "__main__":
    unittest.main()

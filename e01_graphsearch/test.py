import unittest
import ex_graphsearch as ex
import random
import math
from data import graph, sbahn, coordinates


def air_heuristic(u, v):
    air = {'A': {'A': 0, 'B': 15, 'C': 20, 'D': 15},
           'B': {'A': 15, 'B': 0, 'C': 18, 'D': 40},
           'C': {'A': 20, 'B': 18, 'C': 0, 'D': 50},
           'D': {'A': 15, 'B': 40, 'C': 50, 'D': 0}
           }
    return air[u][v]


def sbahn_heuristic(u, v):
    first = coordinates[u]
    second = coordinates[v]
    x1, y1 = float(first[0]), float(first[1])
    x2, y2 = float(second[0]), float(second[1])
    distance = math.sqrt(abs(x1 - x2) + abs(y1 - y2))
    speed = 100.0
    return int(distance/speed)


def check_path(self, path, node):
    p = []
    while node is not None:
        p.append(node.state)
        node = node.parent
    self.assertListEqual(p, path)


def check_fringe(self, data, fringe):
    fringelist = []
    while not fringe.empty():
        fringelist.append(fringe.get().state)
    self.assertListEqual(data, fringelist)


class ExpandTest(unittest.TestCase):
    def check_expand(self, graph, node):
        n = ex.Node(node, None, 0, 0)
        childs = ex.expand(n, graph)
        neighbors = list(graph[node])
        for s in childs:
            neighbors.remove(s.state)
            self.assertEqual(s.parent, n)
            self.assertEqual(s.depth, 1)
            self.assertEqual(s.path_cost, graph[node][s.state])
        self.assertListEqual(neighbors, [])

    def test_small_graph(self):
        self.check_expand(graph, 'A')
        self.check_expand(graph, 'B')
        self.check_expand(graph, 'C')
        self.check_expand(graph, 'D')

    def test_sbahn(self):
        self.check_expand(sbahn, 'Universitaet')
        self.check_expand(sbahn, 'Zuffenhausen')
        self.check_expand(sbahn, 'Marbach (N)')


class DFSTest(unittest.TestCase):
    def test_small_graph(self):
        with self.assertRaises(ex.CutoffException) as cm:
            ex.depth_limited_search(graph, 'A', ['C'], 0)
        node = ex.depth_limited_search(graph, 'D', ['A'], 10)
        check_path(self, ['A', 'D'], node)

    def test_zuffenhausen_universitaet(self):
        node = ex.depth_limited_search(sbahn, 'Zuffenhausen', ['Universitaet'],
                                        12)
        check_path(self, ['Universitaet', 'Schwabstrasse', 'Feuersee',
                          'Stadtmitte', 'Hauptbahnhof', 'Nordbahnhof',
                          'Feuerbach', 'Zuffenhausen', 'Kornwestheim',
                          'Ludwigsburg', 'Kornwestheim', 'Zuffenhausen'], node)

        with self.assertRaises(ex.CutoffException) as cm:
            ex.depth_limited_search(sbahn, 'Zuffenhausen', ['Universitaet'], 6)


class BFSTest(unittest.TestCase):
    def test_small_graph(self):
        node, fringe = ex.breadth_first_search(graph, 'A', ['C'])
        check_path(self, ['C', 'A'], node)
        check_fringe(self, ['D', 'A', 'C'], fringe)

    def test_zuffenhausen_universitaet(self):
        node, fringe = ex.breadth_first_search(sbahn, 'Zuffenhausen',
                                                ['Universitaet'])
        check_path(self, ['Universitaet', 'Schwabstrasse', 'Feuersee',
                          'Stadtmitte', 'Hauptbahnhof', 'Nordbahnhof',
                          'Feuerbach', 'Zuffenhausen'], node)

        self.assertEqual('Oesterreichischer Platz', fringe.get().state)
        self.assertEqual('Rathaus', fringe.get().state)
        self.assertEqual('Berliner Platz', fringe.get().state)
        self.assertEqual('Hauptbahnhof', fringe.get().state)
        self.assertEqual('Feuersee', fringe.get().state)

    def test_vaihingen_hauptbahnhof(self):
        node, fringe = ex.breadth_first_search(sbahn, 'Hauptbahnhof',
                                                ['Vaihingen'])
        check_path(self, ['Vaihingen', 'Oesterfeld', 'Universitaet',
                          'Schwabstrasse', 'Feuersee', 'Stadtmitte',
                          'Hauptbahnhof'], node)

        self.assertEqual('Universitaet', fringe.get().state)
        self.assertEqual('Oesterreichischer Platz', fringe.get().state)
        self.assertEqual('Stadtmitte', fringe.get().state)
        self.assertEqual('Charlottenplatz', fringe.get().state)
        self.assertEqual('Erwin-Schoettle-Platz', fringe.get().state)


class UniformCostTest(unittest.TestCase):
    def test_small_graph(self):
        node, fringe = ex.uniform_cost_search(graph, 'A', ['C'])
        check_path(self, ['C', 'A'], node)
        check_fringe(self, [], fringe)

    def test_zuffenhausen_universitaet(self):
        random.seed(1)
        node, fringe = ex.uniform_cost_search(sbahn, 'Zuffenhausen',
                                               ['Universitaet'])
        check_path(self, ['Universitaet', 'Schwabstrasse', 'Feuersee',
                          'Stadtmitte', 'Hauptbahnhof', 'Nordbahnhof',
                          'Feuerbach', 'Zuffenhausen'], node)

    def test_vaihingen_hauptbahnhof(self):
        random.seed(1)
        node, fringe = ex.uniform_cost_search(sbahn, 'Hauptbahnhof',
                                               ['Vaihingen'])
        check_path(self, ['Vaihingen', 'Oesterfeld', 'Universitaet',
                          'Schwabstrasse', 'Feuersee', 'Stadtmitte',
                          'Hauptbahnhof'], node)


class IterativeDeepeningTest(unittest.TestCase):
    def test_small_graph(self):
        node = ex.iterative_deepening_search(graph, 'A', ['C'])
        check_path(self, ['C', 'A'], node)

    def test_vaihingen_hauptbahnhof(self):
        random.seed(1)
        node = ex.iterative_deepening_search(sbahn, 'Hauptbahnhof',
                                                      ['Vaihingen'])
        check_path(self, ['Vaihingen', 'Oesterfeld', 'Universitaet',
                          'Schwabstrasse', 'Feuersee', 'Stadtmitte',
                          'Hauptbahnhof'], node)

    def test_zuffenhausen_universitaet(self):
        node = ex.iterative_deepening_search(sbahn, 'Zuffenhausen',
                                              ['Universitaet'])
        check_path(self, ['Universitaet', 'Schwabstrasse', 'Feuersee',
                          'Stadtmitte', 'Hauptbahnhof', 'Nordbahnhof',
                          'Feuerbach', 'Zuffenhausen'], node)


class AStarTest(unittest.TestCase):
    def test_small_graph(self):
        node, fringe = ex.a_star_search(graph, 'A', ['C'], air_heuristic)
        check_path(self, ['C', 'A'], node)
        check_fringe(self, ['B', 'D'], fringe)

    def test_vaihingen_hauptbahnhof(self):
        node, fringe = ex.a_star_search(sbahn, 'Hauptbahnhof', ['Vaihingen'],
                                         sbahn_heuristic)
        check_path(self, ['Vaihingen', 'Oesterfeld', 'Universitaet',
                          'Schwabstrasse', 'Feuersee', 'Stadtmitte',
                          'Hauptbahnhof'], node)

    def test_zuffenhausen_universitaet(self):
        node, fringe = ex.a_star_search(sbahn, 'Zuffenhausen', 
                ['Universitaet'], sbahn_heuristic)
        check_path(self, ['Universitaet', 'Schwabstrasse', 'Feuersee',
                          'Stadtmitte', 'Hauptbahnhof', 'Nordbahnhof',
                          'Feuerbach', 'Zuffenhausen'], node)


    def test_zuffenhausen_schwabstrasse_universitaet_vaihingen(self):
        node, fringe = ex.a_star_search(sbahn, 'Zuffenhausen', 
                ['Universitaet','Schwabstrasse', 'Vaihingen'],
                sbahn_heuristic)
        check_path(self, ['Schwabstrasse', 'Feuersee', 'Stadtmitte', 
                          'Hauptbahnhof', 'Nordbahnhof', 'Feuerbach', 
                          'Zuffenhausen'], node)

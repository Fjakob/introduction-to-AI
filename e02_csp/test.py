__author__ = 'johannes'

import ex_csp as ex
import unittest
import numpy as np
import itertools
from data import create_map_csp


class MapColoringTest(unittest.TestCase):
    def setUp(self):
        self.csp = create_map_csp()

    def test_correct_solution_backtracking(self):
        csp_solution = ex.backtracking(self.csp)
        self.assertTrue(csp_solution.complete())

    def test_mrv(self):
        csp_solution, order = ex.minimum_remaining_values(self.csp)
        self.assertTrue(csp_solution.complete())
        print(order)

    def test_mrv_order(self):
        csp_solution, order = ex.minimum_remaining_values(self.csp)
        self.assertEqual(order[0].name, "Schleswig-Holstein")
        self.assertEqual(order[1].name, "Hamburg")
        self.assertEqual(order[2].name, "Niedersachsen")
        self.assertEqual(order[3].name, "Mecklenburg-Vorpommern")
        self.assertEqual(order[4].name, "Brandenburg")
        self.assertEqual(order[5].name, "Sachsen-Anhalt")
        self.assertEqual(order[6].name, "Thueringen")
        self.assertEqual(order[7].name, "Hessen")
        self.assertEqual(order[8].name, "Nordrhein-Westfalen")
        self.assertEqual(order[9].name, "Sachsen")
        self.assertEqual(order[10].name, "Bayern")
        self.assertEqual(order[11].name, "Rheinland-Pfalz")
        self.assertEqual(order[12].name, "Baden-Wuerttemberg")
        self.assertEqual(order[13].name, "Bremen")
        self.assertEqual(order[14].name, "Berlin")
        self.assertEqual(order[15].name, "Saarland")

    def test_mrv_with_degree(self):
        csp_solution, order = \
            ex.minimum_remaining_values_with_degree(self.csp)
        self.assertTrue(csp_solution.complete())
        print(order)

    def test_mrv_with_degree_order(self):
        csp_solution, order = \
            ex.minimum_remaining_values_with_degree(self.csp)
        self.assertEqual(order[0].name, "Niedersachsen")
        self.assertEqual(order[1].name, "Hessen")
        self.assertEqual(order[2].name, "Thueringen")
        self.assertEqual(order[3].name, "Sachsen-Anhalt")
        self.assertEqual(order[4].name, "Brandenburg")
        self.assertEqual(order[5].name, "Bayern")
        self.assertEqual(order[6].name, "Sachsen")
        self.assertEqual(order[7].name, "Mecklenburg-Vorpommern")
        self.assertEqual(order[8].name, "Schleswig-Holstein")
        self.assertEqual(order[9].name, "Nordrhein-Westfalen")
        self.assertEqual(order[10].name, "Rheinland-Pfalz")
        self.assertEqual(order[11].name, "Hamburg")
        self.assertEqual(order[12].name, "Baden-Wuerttemberg")
        self.assertEqual(order[13].name, "Bremen")
        self.assertEqual(order[14].name, "Berlin")
        self.assertEqual(order[15].name, "Saarland")


class SudokuTest(unittest.TestCase):
    def setUp(self):
        with open("sudoku.txt", "r") as f:
            lines = f.readlines()
        sudoku_strs = []
        for line in lines:
            if line[0] == 'G':
                sudoku_strs.append("")
            else:
                sudoku_strs[-1] += line.replace("", " ")[1:]
        self.sudokus = []
        for sudoku_str in sudoku_strs:
            self.sudokus.append(np.fromstring(sudoku_str, sep=' ',
                                              dtype=np.int).reshape((9, 9)))

    def test_sudoku_1(self):
        csp = ex.create_sudoku_csp(self.sudokus[1])
        solution = ex.backtracking(csp)
        sud = ex.sudoku_csp_to_array(solution)
        sudoku_checker(self, sud)

        self.assertTrue(solution.complete())

    def test_sudoku_2(self):
        csp = ex.create_sudoku_csp(self.sudokus[34])
        solution = ex.backtracking(csp)
        sud = ex.sudoku_csp_to_array(solution)
        sudoku_checker(self, sud)

        self.assertTrue(solution.complete())

    def test_sudoku_3(self):
        csp = ex.create_sudoku_csp(self.sudokus[23])
        solution = ex.backtracking(csp)
        sud = ex.sudoku_csp_to_array(solution)
        sudoku_checker(self, sud)

        self.assertTrue(solution.complete())


def all_different(array):
    flat_array = array.reshape((9,))
    for i, j in itertools.combinations(flat_array, 2):
        if i == j:
            return False
    return True


def sudoku_checker(self, sudoku):
    for i in range(9):
        self.assertTrue(all_different(sudoku[i, :]))
        self.assertTrue(all_different(sudoku[:, i]))

    self.assertTrue(all_different(sudoku[0:3, 0:3]))
    self.assertTrue(all_different(sudoku[3:6, 0:3]))
    self.assertTrue(all_different(sudoku[6:9, 0:3]))
    self.assertTrue(all_different(sudoku[0:3, 3:6]))
    self.assertTrue(all_different(sudoku[3:6, 3:6]))
    self.assertTrue(all_different(sudoku[6:9, 3:6]))
    self.assertTrue(all_different(sudoku[0:3, 6:9]))
    self.assertTrue(all_different(sudoku[3:6, 6:9]))
    self.assertTrue(all_different(sudoku[6:9, 6:9]))



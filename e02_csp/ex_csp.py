"""
The exercise is to create a (almost) general constraint satisfaction problem
solver. You will have to use the CSP data structure from csp.py. Read it for
reference!

"""
import numpy as np
import csp
from data import create_map_csp
import itertools


def backtracking(csp, ac_3=False):
    """
    Basic backtracking algorithm to solve a CSP.

    Optional: if ac_3 == True use the AC-3 algorithm for constraint
              propagation. Important: if ac_3 == false don't use
              AC-3!
    :param csp: A csp.ConstrainedSatisfactionProblem object
                representing the CSP to solve
    :return: A csp.ConstrainedSatisfactionProblem, where all Variables
             are set and csp.complete() returns True. (I.e. the solved
             CSP)
    """
    
    if csp.complete():
        return csp
    
    for variable in csp.variables:
        if variable.value is None: #select next unassigned variable
            var = variable
            break
            
    for value in var.domain:
        var.value = value
        if csp.consistent():
            result = backtracking(csp)
            if result is not False:
                return result
        var.value = None
    return False
    

def minimum_remaining_values(csp, order = [], ac_3=False):
    """
    Implement the basic backtracking algorithm to solve a CSP with
    minimum remaining values heuristic and no tie-breaker. Thus the
    first of all best solution is taken.

    Optional: if ac_3 == True use the AC-3 algorithm for constraint
              propagation. Important: if ac_3 == false don't use
              AC-3!
    :param csp: A csp.ConstrainedSatisfactionProblem object
                representing the CSP to solve
    :return: A tuple of 1) a csp.ConstrainedSatisfactionProblem, where
             all Variables are set and csp.complete() returns True. (I.e.
             the solved CSP) and 2) a list of all variables in the order
             they have been assigned.
    """
    if csp.complete():
        return csp, order

    unassigned = []
    for variable in csp.variables:
        if variable.value is None: 
            unassigned.append(variable) #store all unassigned variables
    
    counter = {}  #stores all unassigned variables with legal values
    for variable in unassigned:
        counter[variable.name] = 0
        for value in variable.domain:
            variable.value = value
            if csp.consistent(): #increases the number of legal values of the variable
                counter[variable.name] += 1
            variable.value = None
         
    i = 0
    while i < len(unassigned):
        var = unassigned[i]
        legal = counter[var.name]
        for key in counter:
            if counter[key] < legal:
                nothingSmaller = False
                i += 1
                break
            else: 
                nothingSmaller = True
        if nothingSmaller:
            break

    for value in var.domain:
        var.value = value
        order.append(var)
        if csp.consistent():
            result = minimum_remaining_values(csp, order)
            if result is not False:
                return result
        var.value = None
        order.remove(var)
    return False
    

def minimum_remaining_values_with_degree(csp, order = [], ac_3=False):
    """
    Implement the basic backtracking algorithm to solve a CSP with
    minimum remaining values heuristic and the degree heuristic as
    tie-breaker.

    Optional: if ac_3 == True use the AC-3 algorithm for constraint
              propagation. Important: if ac_3 == false don't use
              AC-3!
    :param csp: A csp.ConstrainedSatisfactionProblem object
                representing the CSP to solve
    :return: A tuple of 1) a csp.ConstrainedSatisfactionProblem, where
             all Variables are set and csp.complete() returns True. (I.e.
             the solved CSP) and 2) a list of all variables in the order
             they have been assigned.
    """
    if csp.complete():
        return csp, order

    unassigned = []
    for variable in csp.variables:
        if variable.value is None: 
            unassigned.append(variable) #stores all unassigned variables
    
    counter = {} #stores tuple of legal values and number of constraints on remainung variables
    for variable in unassigned:
        counter[variable.name] = [0, 0]
        for value in variable.domain:
            variable.value = value
            if csp.consistent(): #increases the number of legal values of the variable
                counter[variable.name][0] += 1
            variable.value = None
        for peer in variable.peers:
            if peer.value is None:
                counter[variable.name][1] += 1 #sets number of constraints on remaining variables
        
    i = 0
    while i < len(unassigned):
        var = unassigned[i]
        legal, remainConstr = counter[var.name]
        for key in counter:
            if counter[key][0] < legal:
                i += 1
                nothingBetter = False
                break
            elif counter[key][0] == legal:
                if counter[key][1] > remainConstr:
                    i += 1
                    nothingBetter = False
                    break
                else: nothingBetter = True
            else: nothingBetter = True
        if nothingBetter:
            break

    for value in var.domain:
        var.value = value
        order.append(var)
        if csp.consistent():
            result = minimum_remaining_values_with_degree(csp, order)
            if result is not False:
                return result
        var.value = None
        order.remove(var)
    return False


def create_sudoku_csp(sudoku):
    """
    Creates a csp.ConstrainedSatisfactionProblem from a numpy array
    `sudoku` which has shape (9, 9). Each entry of the sudoku is either
    0, which means it is not set yet or in [1, ..., 9], which means
    it is already assigned a number.

    The CSP should contain all constraints necessary to solve the sudoku.
    I.e. no two numbers in a row must be equal, no two numbers in a column
    must be equal and no two numbers in one of the 9 3x3 blocks must be
    equal. All numbers in the array must be already set.

    :param sudoku: A numpy array representing a unsolved sudoku
    :return: A csp.ConstrainedSatisfactionProblem which can be used
             to solve the sudoku
    """
    variables, constraints, domain = [], [], [1, 2, 3, 4, 5, 6, 7, 8, 9]
    i = 1
    for row in sudoku:
        ascii = 65
        for entry in row:
            if entry == 0:
                variables.append(csp.Variable(str(i) + chr(ascii), domain, None))
            else:
                variables.append(csp.Variable(str(i) + chr(ascii), domain, entry))
            ascii += 1
        i += 1
    
    #all row constraints
    a, b = 0, 9
    for k in range(9):
        for i, j in itertools.combinations(variables[a:b], 2):
            constraints.append(csp.UnequalConstraint(i, j))
        a = b
        b += 9
    
    #all column constraints
    columnList = []
    a, i = 0, 0
    for l in range(9):
        column = []
        for k in range(9):
            column.append(variables[a])
            a += 9
        columnList.append(column)
        i += 1
        a = i
        
    for column in columnList:
        for i, j in itertools.combinations(column, 2):
            constraints.append(csp.UnequalConstraint(i, j))
    
    #all block constraints
    a, b = 0, 0
    block, blocklist = [], []
    for m in range(3):
        for l in range(3):
            for k in range(3):
                for i in range(81)[b:b+3]:
                    i = i+a
                    block.append(variables[i])
                a = a+9
            b += 3
            blocklist.append(block)
            block = []
            a=0
        b += 18 
        
    for block in blocklist:
        for i, j in itertools.combinations(block, 2):
            add = True
            for added in constraints:
                if (added.var1.name is i.name and added.var2.name is j.name) or (added.var1.name is j.name and added.var2.name is i.name):
                    add = False
            if add:
                constraints.append(csp.UnequalConstraint(i, j))  
    
    return csp.ConstrainedSatisfactionProblem(variables, constraints)


def sudoku_csp_to_array(csp):
    """
    Takes a sudoku CSP from `create_sudoku_csp()` as you implemented
    it and returns a numpy array s with `s.shape == (9, 9)` (i.e. a
    9x9 matrix) representing the sudoku.

    :param csp: The CSP created with `create_sudoku_csp()`
    :return: A numpy array with shape (9, 9)
    """

    sudoku = np.zeros((9,9))
    k = 0
    for j in range(9):
        for i in range(9):
            if csp.variables[k].value is None:
                sudoku[j][i] = 0
            else: 
                sudoku[j][i] = csp.variables[k].value
            k += 1
        
    return sudoku


def read_sudokus():
    """
    Reads the sudokus in the sudoku.txt and saves them as numpy arrays.
    :return: A list of np.arrays containing the sudokus
    """
    with open("sudoku.txt", "r") as f:
        lines = f.readlines()
    sudoku_strs = []
    for line in lines:
        if line[0] == 'G':
            sudoku_strs.append("")
        else:
            sudoku_strs[-1] += line.replace("", " ")[1:]
    sudokus = []
    for sudoku_str in sudoku_strs:
        sudokus.append(np.fromstring(sudoku_str, sep=' ',
                                     dtype=np.int).reshape((9, 9)))
    return sudokus


def main():
    """
    A main function. This might be useful for developing, if you don't
    want to run all tests all the time. Just write here what ever you
    want to develop your code. If you use pycharm you can run the unittests
    also by right-clicking them and then e.g. "Run 'Unittest test_sudoku_1'".
    """
    
    # first lets test with a already created csp:
    csp = create_map_csp()
    solution = backtracking(csp)
    print(solution)
    
    # and now with our own generated sudoku CSP
    sudokus = read_sudokus()
    csp = create_sudoku_csp(sudokus[1])
    print sudoku_csp_to_array(csp), "\n"
    solution, order = minimum_remaining_values_with_degree(csp)
    print sudoku_csp_to_array(solution), "\n"
    
    
if __name__ == '__main__':
    main()
    
    
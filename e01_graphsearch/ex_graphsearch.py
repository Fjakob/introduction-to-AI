from data import graph, sbahn, coordinates
from Queue import Queue, LifoQueue, PriorityQueue


class FailureException(Exception):
    """
    An exception that is raised when a failure occurs
    """
    pass


class CutoffException(Exception):
    """
    An exception, that is raised when a cutoff appears
    """
    pass

class UnvalidGoalException(Exception):
    """
    An exception, that is raised when the graph does not
    contain one of the given goals
    """
    pass


class Node(object):
    """
    A very simple node data structure for Trees.
    """
    def __init__(self, state, parent, path_cost, depth):
        self.state = state

        self.parent = parent
        self.children = []
        self.path_cost = path_cost
        self.depth = depth

        if self.parent is not None:
            self.parent.children.append(self)

    def __str__(self):
        string = "'" + self.state + "'"
        if self.parent is not None:
            string = string + ", " + str(self.parent)
        return string


class NodePriorityQueue(object):
    """
    A PriorityQueue that uses the path_cost as priority. When a heuristic and
    a goal state is given the heuristic cost from each node to the goal state
    is added to the priority.
    """
    def __init__(self, heuristic=None, goals=[None]):
        self.visited = []
        self.queue = PriorityQueue()
        self.goals = goals

        if heuristic is not None:
            self.heuristic = heuristic
        else:
            # if no heuristic is given use a constant 0 heuristic
            self.heuristic = lambda u, v: 0

    def put(self, node):
        """
        This function puts a node with the correct priority into the
        internal priority queue. You don't have to take care to adjust
        earlier added nodes
        """
        if self.heuristic is not None:
            nearest_goal = self.goals[0]
            for goal in self.goals:
                if self.heuristic(node.state, goal) < self.heuristic(node.state, nearest_goal):
                    nearest_goal = goal
            priority = node.path_cost + self.heuristic(node.state, nearest_goal)
        else:
            priority = node.path_cost
        
        self.queue.put((priority, node))

    def get(self):
        """
        This function takes care that already visited nodes are not visited
        again.
        """
        _, node = self.queue.get()
        while node.state in self.visited:
            _, node = self.queue.get()

        self.visited.append(node.state)
        return node

    def empty(self):
        """
        empty() returns true if the queue is empty or has only already
        visited nodes in it.
        """
        if self.queue.empty():
            return True
        priority, node = self.queue.get()
        while node.state in self.visited:
            if self.queue.empty():
                return True
            priority, node = self.queue.get()

        self.queue.put((priority, node))
        return False


def expand(node, graph):
    """
    Expands a node in a given graph
    
    :param graph: The graph, that defines the problem
    :return: A list of all successors
    """
    successors = []
    
    for neighbour in graph[node.state]:
        n = Node(neighbour, node, node.path_cost + graph[node.state][neighbour], node.depth + 1)
        successors.append(n)
    
    return successors
 

def breadth_first_search(graph, start, goals):
    """
    TreeSearch that searchs by looking for the goal in the breadth first

    :param graph: The graph to run the BFS on
    :param start: The start node
    :param goals: A list of nodes that should be reached (one of them)
    :return: a tuple with a the found node as first entry and the
            fringe as the second entry
    """
    for goal in goals:
        if goal not in graph:
            raise UnvalidGoalException
    
    if type(start) is not Node:
        start = Node(start, None, 0, 0)
        
    fringe = Queue()
    fringe.put(start)
    for goal in goals:
            if goal == start.state: return (start, fringe)
       
    while True:
        if fringe is None: raise FailureException
        node = fringe.get()
        for goal in goals:
            if goal == node.state: return node, fringe
        for successor in expand(node, graph):
            fringe.put(successor)
                           
                
def uniform_cost_search(graph, start, goals):
    """
    TreeSearch that searchs by looking for the goal in the breadth with
    the minimum path cost first

    :param graph: The graph to run the BFS on
    :param start: The start node
    :param goals: A list of nodes that should be reached (one of them)
    :return: a tuple with a the found node as first entry and the
            fringe as the second entry
    """
    for goal in goals:
        if goal not in graph:
            raise UnvalidGoalException
    
    if type(start) is not Node:
        start = Node(start, None, 0, 0)
        
    fringe = NodePriorityQueue()
    fringe.put(start)
       
    while True:
        if fringe is None: raise FailureException
        node = fringe.get()
        for goal in goals:
            if goal == node.state: return node, fringe
        for successor in expand(node, graph):
            fringe.put(successor)
            
        
def depth_limited_search(graph, start, goals, limit):
    """
    Recursive depth-limited-search.
    Raises a CutoffException if the goal could not by found due to the limit

    :param graph: The graph to run the DFS on
    :param start: The start node
    :param goals: A list of nodes that should be reached (one of them)
    :param limit: The depth limit
    :return: the found node
    """
    for goal in goals:
        if goal not in graph:
            raise UnvalidGoalException
    
    if type(start) is not Node:
        start = Node(start, None, 0, 0)
    node = start
        
    for goal in goals:
        if goal == node.state: return node
    if node.depth == limit:
        raise CutoffException
    else: 
        for successor in expand(node, graph):
            try:
                result = depth_limited_search(graph, successor, goals, limit)
                return result
            except(CutoffException):
                cutoff = True
        if cutoff: raise CutoffException
            
    
def iterative_deepening_search(graph, start, goals):
    """
    Recursive depth-limited-search with increasing limit

    :param graph: The graph to run the DFS on
    :param start: The start node
    :param goals: A list of nodes that should be reached (one of them)
    :return: the found node
    """
    for goal in goals:
        if goal not in graph:
            raise UnvalidGoalException
    
    limit = 0
    while True:
        try:
            limit = limit + 1
            node = depth_limited_search(graph, start, goals, limit)
            return node
        except(CutoffException):
            pass


def a_star_search(graph, start, goals, heuristic):
    """
    A-star-search which looks for nodes with minimum path cost and
    minimum heuristic value

    :param graph: The graph to run the DFS on
    :param start: The start node
    :param goals: A list of nodes that should be reached (one of them)
    :param heuristic: A function that returns a heuristic cost value
                      for two given nodes. E.g. heuristic('A', 'B')
                      returns a float
    :return: a tuple with the found node as first entry and the
            fringe as the second entry
    """
    for goal in goals:
        if goal not in graph:
            raise UnvalidGoalException
    
    if type(start) is not Node:
        start = Node(start, None, 0, 0)
        
    fringe = NodePriorityQueue(heuristic, goals)
    fringe.put(start)
    
    nearest_goal = goals[0]
    for goal in goals:
        if heuristic(start.state, goal) < heuristic(start.state, nearest_goal):
            nearest_goal = goal

    while True:
        if fringe is None: raise FailureException
        node = fringe.get()
        if nearest_goal is node.state: return node, fringe
        for successor in expand(node, graph):
            fringe.put(successor)
            
            

  
if __name__ == '__main__':
    # main method
    
    try:
        node, fringe = uniform_cost_search(sbahn, "Vaihingen", ["Waiblingen"])
        liste = []
        while not fringe.empty(): liste.append(fringe.get().state)
        print "Fastest path:\n", node
        print "\nThe path takes round about {} minutes\n".format(node.path_cost)
        print "Fringe:\n", liste
        node = iterative_deepening_search(graph, "D", ["C"])
        print "\n", node
    except(UnvalidGoalException):
        print "Goal is not existent!"
    except(CutoffException):
        print "Goal is beyond limit!"
    except(FailureException):
        print "Failure!"

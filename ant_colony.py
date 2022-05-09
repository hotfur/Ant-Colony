# We can use CuPy to accelerate the calculation of matrices
import numpy as np

"""
Ant colony algorithm
@author Vuong Kha Sieu
@date 8/5/2022
"""


class Graph:
    """
    A class for codifying graphs to solve the TSP with Ant Colony algorithm.
    One constructor for transforming the adjacency matrix to adjacency list and edge list. Three methods: a travelling
    ant function to deposits pheromone, a update pheromone function for updating pheromone by several ants and a
    harvester function for collecting the best trails left behind.
    """
    def __init__(self, weight_matrix, alpha, beta, phi, init, qui):
        """
        Initialize a graph from its weight matrix and several constants for calculation of pheromone deposition and ant
        path selection probability.
        :param weight_matrix: the weight matrix, must contains only zeros on the diagonal and for
        edges that do not exist
        :param alpha: constant adjusting the power of pheromone in selection of trails
        :param beta: constant adjusting the power of edge weight in selection of trails
        :param phi: pheromone evaporation constant, has to be a float and smaller than 1
        :param init: the pheromone value to initialize the pheromone matrix, has to be a float
        :param qui: constant adjusting the power of edges' weight in updating the pheromone deposition
        :return: an object which contains the inverted weight matrix, a pheromone matrix initialized with same pheromone
        level, an edge list and an adjacency list for faster calculation in other methods.
        """
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.Qui = qui
        # Create an edge list
        shape = np.shape(weight_matrix)
        edge_list = np.argwhere(weight_matrix > 0)
        edge_list.sort()
        edge_list = np.unique(edge_list, axis=0)
        # Create an adjacency list
        adj_list = dict()
        for edge in edge_list:
            if edge[0] not in adj_list:
                adj_list[edge[0]] = []
            if edge[1] not in adj_list:
                adj_list[edge[1]] = []
            adj_list[edge[0]].append(edge[1])
            adj_list[edge[1]].append(edge[0])
        # Calculate the invert of the weights for assignment to the new weight matrix
        self.weight_matrix = np.power(weight_matrix, -1)
        # Initialize the pheromone matrix
        self.phero_matrix = np.where(weight_matrix != 0, float(init), 0)
        self.edge_list = edge_list
        self.adj_list = adj_list
        self.verticles_no = shape[0]
        self.iteration = 0

    def update_pheromone(self, phero_changes):
        """Update pheromone function

        :argument phero_changes: a dictionary containing edges in which ants has travelled in their tours
        :return: nothing
        """
        self.phero_matrix = np.where(self.phero_matrix != 0, self.phero_matrix*(1 - self.phi), 0)
        for i in phero_changes:
            self.phero_matrix[i[0]][i[1]] += phero_changes[i] * self.Qui * self.weight_matrix[i[0]][i[1]]
            self.phero_matrix[i[1]][i[0]] += phero_changes[i] * self.Qui * self.weight_matrix[i[1]][i[0]]
        self.iteration += 1
        return

    def ant_harvester(self):
        """Harvest the TSP optimal solution by ant colony

        :return: a sorted list of edges by pheromone that make up the best tour"""
        result = []
        for i in self.edge_list:
            result.append((i[0], i[1], self.phero_matrix[i[0]][i[1]]))
        result.sort(reverse=True, key=lambda y: y[2])
        result = result[:self.verticles_no]
        return result


def traveling_ant(graph, init_post, phero_changes, visited):
    """
    Recursive traveling ant function to deposits pheromone

    :argument init_post: the ant starting vertex
    :argument phero_changes: the pheromone changes dictionary
    :argument visited: the visited vertices list
    :returns: empty pheromone change dictionary if the ant failed to complete a tour, or a complete pheromone
    changes dictionary for successful tours
    """
    visited.append(init_post)
    if len(visited) == graph.verticles_no:
        # Base case 1: exist an edge between the last and first vertices in the tour
        if graph.phero_matrix[init_post][visited[0]] != 0:
            if init_post < visited[0]:
                phero_changes[(init_post, visited[0])] = 1
            else:
                phero_changes[(visited[0], init_post)] = 1
            return phero_changes
        # Base case 2: All vertex has been visited, but there is no way to return to the starting point.
        return dict()

    # Eliminate all visited vertices from the list of possible candidates for the ant to visit
    neighbors = list(graph.adj_list[init_post])
    for vertex in visited:
        try:
            neighbors.remove(vertex)
        except:
            continue
    # Base case 3: only a fraction of the vertex has been visited and there is no way to continue moving unless we
    # visit a vertex more than once.
    if len(neighbors) == 0:
        return dict()

    # Recursive case
    else:
        # This array stores the chance that the ant will visits its neighbors
        chance = np.empty(len(neighbors))
        random = np.random.rand(len(neighbors))
        for i in range(len(neighbors)):
            chance[i] = (graph.phero_matrix[init_post][neighbors[i]] ** graph.alpha) * \
                        (graph.weight_matrix[init_post][neighbors[i]] ** graph.beta)
        chance_sum = np.sum(chance)
        chance = chance * random / chance_sum
        next_move = neighbors[int(np.where(chance == np.amax(chance))[0])]
        # We want the edges to be in order where first vertex is always the smaller numbered one
        if init_post < next_move:
            phero_changes[(init_post, next_move)] = 1
        else:
            phero_changes[(next_move, init_post)] = 1
        return traveling_ant(graph, next_move, phero_changes, visited)


def travelling_ant_warpper(graph, init_position):
    """
    A wrapper for calling travelling ant function with only 2 parameters
    :param graph: the graph to operate on
    :param init_position: the initial position of the ant
    :return: empty pheromone change dictionary if the ant failed to complete a tour, or a complete pheromone
    changes dictionary for successful tours
    """
    return traveling_ant(graph, init_position, dict(), list())

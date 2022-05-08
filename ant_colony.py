"""
Ant colony algorithm
@author Vuong Ksa Sieu
@date 8/5/2022
"""
import numpy as np

"""Init, phi and all weights has to be float"""
class Graph:
    def __init__(self, weight_matrix, alpha, beta, phi, init, qui):
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.Qui = qui
        shape = np.shape(weight_matrix)
        adj_list = dict()
        edge_list = np.argwhere(weight_matrix>0)
        edge_list.sort()
        edge_list = np.unique(edge_list, axis=0)
        for edge in edge_list:
            if edge[0] not in adj_list:
                adj_list[edge[0]] = []
            if edge[1] not in adj_list:
                adj_list[edge[1]] = []
            adj_list[edge[0]].append(edge[1])
            adj_list[edge[1]].append(edge[0])
        self.weight_matrix = np.power(weight_matrix, -1)
        self.phero_matrix = np.where(weight_matrix != 0, float(init), 0)
        self.edge_list = edge_list
        self.adj_list = adj_list
        self.verticles_no = shape[0]
        self.iteration = 0

    """
    Traveling ants
    """

    def traveling_ant(self, init_post, phero_changes, visited):
        visited.append(init_post)
        if len(visited) == self.verticles_no:
            return phero_changes

        neighbors = list(self.adj_list[init_post])
        for vertex in visited:
            try:
                neighbors.remove(vertex)
            except:
                continue
        if len(neighbors) == 0:
            return dict()
        else:
            chance = np.empty(len(neighbors))
            random = np.random.rand(len(neighbors))
            for i in range(len(neighbors)):
                chance[i] = (self.phero_matrix[init_post][neighbors[i]] ** self.alpha) * \
                            (self.weight_matrix[init_post][neighbors[i]] ** self.beta)
            chance_sum = np.sum(chance)
            chance = chance * random / chance_sum
            next_move = neighbors[int(np.where(chance == np.amax(chance))[0])]
            if init_post < next_move:
                phero_changes[(init_post, next_move)] = self.Qui * self.weight_matrix[init_post][next_move]
            else:
                phero_changes[(next_move, init_post)] = self.Qui * self.weight_matrix[init_post][next_move]
            return self.traveling_ant(next_move, phero_changes, visited)

    """Update pheromone function"""

    def update_pheromone(self, phero_changes):
        self.phero_matrix *= (1 - self.phi)
        for i in phero_changes:
            self.phero_matrix[i[0]][i[1]] += phero_changes[i]
            self.phero_matrix[i[1]][i[0]] += phero_changes[i]
        self.iteration += 1
        if len(self.edge_list) <= self.verticles_no:
            return 0
        return 1

    """Harvest the TSP optimal solution by ant colony"""

    def ant_harvester(self):
        result = []
        for i in self.edge_list:
            result.append((i[0], i[1], self.phero_matrix[i[0]][i[1]]))
        result.sort(reverse=True, key=lambda y: y[2])
        result = result[:self.verticles_no]
        return result


def ant_solver(graph, iteration):
    while graph.iteration < iteration:
        phero_changes = dict()
        for i in range(graph.verticles_no):
            temp_changes = graph.traveling_ant(i, dict(), list())
            for k in temp_changes:
                if k in phero_changes:
                    phero_changes[k] += temp_changes[k]
                else:
                    phero_changes[k] = temp_changes[k]
        graph.update_pheromone(phero_changes)
    return graph.ant_harvester()

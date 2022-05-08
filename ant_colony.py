"""
Ant colony algorithm
@author Vuong Ksa Sieu
@date 8/5/2022
"""
import numpy as np

class Graph:
    def __init__(self, weight_matrix, alpha, beta, phi, init, Qui):
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.Qui = Qui
        shape = np.shape(weight_matrix)
        phero_matrix = np.zeros(shape)
        edge_list = []
        adj_list = dict()
        for x in range(shape[0]):
            for y in range(x+1, shape[0]):
                if weight_matrix[x][y] != 0:
                    if x not in adj_list:
                        adj_list[x] = []
                    if y not in adj_list:
                        adj_list[y] = []
                    adj_list[x].append(y)
                    adj_list[y].append(x)
                    phero_matrix[x][y] = init
                    phero_matrix[y][x] = init
                    edge_list.append((x, y))
        self.weight_matrix = np.power(weight_matrix, -1)
        self.phero_matrix = phero_matrix
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

        neighbors = self.adj_list[init_post]
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
            chance_sum = 0
            for i in range(len(neighbors)):
                chance[i] = (self.phero_matrix[init_post][neighbors[i]]**self.alpha) * \
                            (self.weight_matrix[init_post][neighbors[i]]**self.beta)
                chance_sum += chance[i]
            for i in range(len(neighbors)):
                chance[i] = (chance[i]*random[i])/chance_sum
            next_move = neighbors[int(np.where(chance == np.amax(chance))[0])]
            if init_post < next_move:
                phero_changes[(init_post, next_move)] = self.Qui * self.weight_matrix[init_post][next_move]
            else:
                phero_changes[(next_move, init_post)] = self.Qui * self.weight_matrix[init_post][next_move]
            return self.traveling_ant(next_move, phero_changes, visited)

    """Update pheromone function"""
    def update_pheromone(self, phero_changes):
        for i in self.edge_list:
            self.phero_matrix[i[0]][i[1]] *= (1 - self.phi)
            self.phero_matrix[i[1]][i[0]] *= (1 - self.phi)
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
        #result = result[:self.verticles_no]
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


import ant_colony
import numpy as np

matrix = np.array([[0.,7.,8.,9.],[7.,0.,6.,8.],[8.,6.,0.,2.],[9.,8.,2.,0.]])
graph01 = ant_colony.Graph(matrix, 2, 1.1, 0.1, 3, 5)

edges = ant_colony.ant_solver(graph01, 4)
print(edges)
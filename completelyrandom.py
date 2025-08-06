# This does not require dataa and will create and entirely randomized graph connected via Delauney Triangulation.

import sys
import numpy as np

from main import main
from classes.TwoDGraph import TwoDGraph
from scipy.spatial import Delaunay

def creategraph(vertexnumber):
    #Step one: create random number of vertices on the inside, done via dirichlet distribution (Idk either) and barycentric coordinates of the hull

    newvertices = np.zeros((vertexnumber, 2))
    for i in range(vertexnumber):
        x = np.random.rand()
        y = np.random.rand()
        newvertices[i,0] = x
        newvertices[i,1] = y

    #Step two: create the graph based of these vertices with Delaunay
    tri = Delaunay(newvertices)
    return TwoDGraph(vertices=newvertices.T, faces=tri.simplices)


if __name__ == '__main__':
    #we first randomize the state. This is ironically so we can revisit graphs later on by reusing the seed because the get_state function is giving me an aneurysm
    seed = int(200 * np.random.rand())
    np.random.seed(seed)

    vertexnumber = int(3 + 10*np.random.rand())
    print(vertexnumber)

    randomized_graph = creategraph(vertexnumber)

    #for naming convention, I will use the state of the random numpy generator that generates our graph
    main(randomized_graph, "data/2dfolder/randomized/Graph_of_seed_" + str(seed) + ".txt")

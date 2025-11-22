# This is for data where only a hull is provided. The points inside the hull will be randomized and then connected via Delauney Triangulation. 

import numpy as np
import sys
sys.path.insert(0, 'C:/Users/ich/Desktop/Uni/Alternating-Edge-Sum')
sys.path.insert(0, 'C:/Users/daveb/OneDrive/Desktop/Uni/Alternating-Edge-Sum/')


from main import main
from util.TwoDGraph import TwoDGraph
from scipy.spatial import Delaunay

filepath = "data/2dfolder/onlyhulls/basichull.txt"
outputfolder = "output/2dfolder/testdata1811/"

attempts = 1
stepsize = 500000

#How many randomization of one file do you want?
tries = 1

def creategraphfromhull(hull: np.ndarray, nrinsides: int):
    #Step one: create random number of vertices on the inside, done via dirichlet distribution (Idk either) and barycentric coordinates of the hull
    hullsize = hull.shape[0]
    innervertices = []

    for i in range(nrinsides):
        parameters = np.random.dirichlet(np.ones(hullsize),size=1)
        newvertex = np.matmul(parameters, hull)
        innervertices.append(newvertex)

    realinnervertices = np.squeeze(np.array(innervertices))
    fullvertices = np.concatenate((hull, realinnervertices), axis = 0)

    #Step two: create the graph based of these vertices with Delaunay
    tri = Delaunay(fullvertices)
    return TwoDGraph(vertices=fullvertices.T, faces=tri.simplices.tolist())


if __name__ == '__main__':
    # if(len(sys.argv)<2):
    #     raise Exception("You need to provide Graph Data")
    # filepath = sys.argv[1]

    #get hull vertices, should have format x1 y1\n x2 y2 etc.
    for i in range(tries):
        max_vertices = 60
        vertices = np.loadtxt(filepath, delimiter = " ")
        seed = 163#int(200 * np.random.rand())
        np.random.seed(seed)
        nrinsides = int(2 + max_vertices*np.random.rand())

        randomized_graph = creategraphfromhull(vertices, nrinsides)
        main(randomized_graph, outputfolder + "basichull" + str(seed) + "_" + str(max_vertices), attempts=attempts, stepsize=stepsize)






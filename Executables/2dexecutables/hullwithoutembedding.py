# This is for data where only a hull is provided. The points inside the hull will be randomized and then connected via Delauney Triangulation. The insides will then also be randomized to destory the embedding

import numpy as np
import sys
sys.path.insert(0, 'C:/Users/ich/Desktop/Uni/Alternating-Edge-Sum')
sys.path.insert(0, 'C:/Users/daveb/OneDrive/Desktop/Uni/Alternating-Edge-Sum/')


from main import main
from util.TwoDGraph import TwoDGraph
from util.Graphutil import fixorientation
from scipy.spatial import Delaunay

filepath = "data/2dfolder/onlyhulls/basichull.txt"
outputfolder = "output/2dfolder/testdatanoembedding/"

attempts = 1
stepsize = 1000000

#number of random graphs you wanna generate this way
tries = 3

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

    #Step three: Shuffle the vertices on the inside to destroy the embedding. Then combine with hull to have all the vertices
    shuffleseed = 107#int(200*np.random.rand()) # So we can reuse graphs with different shuffling
    np.random.seed(shuffleseed)
    neworder = np.arange(nrinsides)
    np.random.shuffle(neworder) # new order of the vertices
    fullvertices = np.concatenate((hull, realinnervertices[neworder,:]), axis = 0)

    return TwoDGraph(vertices=fullvertices.T, faces=tri.simplices.tolist()), shuffleseed


if __name__ == '__main__':
    # if(len(sys.argv)<2):
    #     raise Exception("You need to provide Graph Data")
    # filepath = sys.argv[1]

    #get hull vertices, should have format x1 y1\n x2 y2 etc.
    vertices = np.loadtxt(filepath, delimiter = " ")
    maxvertices = 400
    for i in range (tries):
        seed = int(200 * np.random.rand())
        np.random.seed(seed)
        nrinsides = int(2 + maxvertices*np.random.rand())

        randomized_graph, shuffleseed = creategraphfromhull(vertices, nrinsides)
        #real_graph = fixorientation(randomized_graph)

        #for naming convention, I will use the state of the random numpy generator that generates our graph
        main(randomized_graph, outputfolder + "noembedding_" + str(seed) + "_" + str(shuffleseed) + "_" + str(nrinsides) + "_outta_" + str(maxvertices), attempts, stepsize)

        #repeat with new shuffling
        randomized_graph2, shuffleseed2 = creategraphfromhull(vertices, nrinsides)

        main(randomized_graph2, outputfolder + "noembedding_" + str(seed) + "_" + str(shuffleseed2) + "_" + str(nrinsides) + "_outta_" + str(maxvertices), attempts, stepsize)




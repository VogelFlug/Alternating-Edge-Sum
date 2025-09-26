#collection of functions that didn't fit anywhere else and aren't too important for us anymore

import numpy as np
import torch

from . import Graphutil as gutil
from .TwoDGraph import TwoDGraph

#Tutteembedding via Gradient Descent
def gradienttutte(Graph: TwoDGraph, learnrate: float):
    '''Gradient descent to achieve Tutte embedding. Not important, simply an example to practice gradient descent'''
    vertices = Graph.vertices
    oe = gutil.getouteredges(Graph.edgecounter)
    ie = gutil.getinneredges(Graph.edgecounter)

    ov = gutil.getoutervertices(oe)
    iv = gutil.getinnervertices(Graph.vertexnumber, ov)

    Xtensor = torch.tensor((vertices[0,iv]).tolist(), requires_grad=True)
    Ytensor = torch.tensor((vertices[1,iv]).tolist(), requires_grad=True)

    for i in range(2000):
        #calc energy
        xenergy = tutteenergy(Xtensor, vertices, iv, Graph.neighbourhood, 0)
        yenergy = tutteenergy(Ytensor, vertices, iv, Graph.neighbourhood, 1)
        
        #get gradient through backpropagation
        xenergy.backward() # type: ignore
        yenergy.backward() # type: ignore
        #print(Xtensor)
        #print(Xtensor.grad)
        #print(Ytensor.grad)
        with torch.no_grad():
            Xtensor -= learnrate * Xtensor.grad # type: ignore
            Ytensor -= learnrate * Ytensor.grad # type: ignore

        #print(Xtensor)
        #Gradienten zurücksetzen
        Xtensor.grad.zero_() # type: ignore
        Ytensor.grad.zero_() # type: ignore
    
    vertexs = np.zeros((2,Graph.vertexnumber))
    vertx = Xtensor.detach().numpy()
    verty = Ytensor.detach().numpy()
    for i in range(Graph.vertexnumber):
        if(i in iv):
            vertexs[0,i] = vertx[iv.index(i)]
            vertexs[1,i] = verty[iv.index(i)]
        else:
            vertexs[:,i] = Graph.vertices[:,i]

    newGraph = TwoDGraph(vertices=vertexs, faces=Graph.faces)
    return newGraph



def tutteenergy(innervertices, allvertices: np.ndarray, innervertexindices, neighbourhood: tuple[set[int]], axis):
    '''Calculates the "Tutte energy" aka how close to a Tutte embedding a Graph is for the given axis. 
    Not particularly important for the problem at hand, just serves as a simple example of gradient descent

    # Input Variables:
    # innervertices = the tensor we are trying to optimize, holding the coordinates of the innervertices
    # allvertices = all of the vertices in the Graph
    # innervertexindices = a list that holds the indices of the innervertices tensor within the allvertices array (aka the actual indices of the inner vertices within the graph)
    # neighbourhood = a tuple of sets, each set holding the neighbourhood of the vertex at that index
    # axis = the axis over which you wish to calculate this energy

    # Output:
    # A squared sum over all vertices of how close each of them is to being the average of each of his neighbours for that specific axis
    '''
    energy = 0
    for i,vertex in enumerate(innervertices):
        nh = tuple(neighbourhood[innervertexindices[i]])
        #print(nh)
        neighbourcounter = len(nh)
        neighbourhoodsum = 0
        for neighbour in nh:
            if(neighbour in innervertexindices):
                neighbourhoodsum += innervertices[innervertexindices.index(neighbour)]
            else:
                neighbourhoodsum += allvertices[axis, neighbour]
        energy += (vertex-(neighbourhoodsum/neighbourcounter))**2

    #graphy = torchviz.make_dot(energy, params={v: k for v, k in enumerate(list(innervertices))})
    #graphy.render("scuffedaaahfile", format="png")
    return energy
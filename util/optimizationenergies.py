# This file holds all the different energy functions used throughout different files
import numpy as np
import torch

from . import Graphutil as gutil
from .TwoDGraph import TwoDGraph

def SnapshotAES(Graph: TwoDGraph):
    '''Calculates the AES energy for a given graph, Might be scuffed since implementation is copied from regular AESenergy. 
    This had a TODO: fix statement on it that i never removed, so there might be something wrong with it
    '''

    ie = gutil.getinneredges(Graph.edgecounter)

    aeslist = gutil.getAESList(Graph, ie)
    fullvertex = Graph.vertices
    
    i = fullvertex[:,aeslist[:,0]]
    k = fullvertex[:,aeslist[:,1]]
    j = fullvertex[:,aeslist[:,2]]
    l = fullvertex[:,aeslist[:,3]]

    #energy for each inner edge = (|ik| - |kj| + |jl| - |li|) ^ 2
    energies = (np.linalg.norm(k-i, axis=0) - np.linalg.norm(j-k, axis=0) + np.linalg.norm(l-j, axis=0) - np.linalg.norm(i-l, axis=0))
    
    energy = np.sum(energies ** 2)

    return np.sum(energies ** 2)


def AESvertexenergy(innervertices, allvertices: np.ndarray, innervertexindices, aeslist: np.ndarray):
    '''Calculates the AES energy of a Graph (same as above, but with a tensor for gradient descent) with fixed outter vertices and 
    using the vertex positions as the variables to be optimized

    # Input Variables:
    # innervertices = the tensor we are trying to optimize, holding the coordinates of the innervertices
    # allvertices = all of the vertices in the Graph
    # innervertexindices = a list that holds the indices of the innervertices tensor within the allvertices array (aka the actual indices of the inner vertices within the graph)
    # aeslist = list where each row corresponds to one inner edge (i,j) and its adjacent faces with format [i, l, j, k]
    '''
    fullvertex = torch.tensor(allvertices.tolist(), requires_grad=False)
    #This combines our fixed outside vertices with the changing inner vertices
    for count, i in enumerate(innervertexindices):
        fullvertex[:,i] = innervertices[:,count]
    
    i = fullvertex[:,aeslist[:,0]]
    k = fullvertex[:,aeslist[:,1]]
    j = fullvertex[:,aeslist[:,2]]
    l = fullvertex[:,aeslist[:,3]]

    #energy for each inner edge = (|ik| - |kj| + |jl| - |li|) ^ 2
    energiez = (torch.linalg.norm(k-i, dim=0) - torch.linalg.norm(j-k, dim=0) + torch.linalg.norm(l-j, dim=0) - torch.linalg.norm(i-l, dim=0))
    
    energy = torch.sum(energiez ** 2)

    # graphy = torchviz.make_dot(energy, params={v: k for v, k in enumerate(list(innervertices))})
    # graphy.render("AEStree", format="png")
    return torch.sum(energiez ** 2)


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
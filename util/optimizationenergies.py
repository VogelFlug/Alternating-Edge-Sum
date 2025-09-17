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


def AESinnervertexenergy(innervertices, allvertices: np.ndarray, innervertexindices, aeslist: np.ndarray):
    '''Calculates the AES energy of a Graph (same as above, but with a tensor for gradient descent) with fixed outter vertices and 
    using the vertex positions as the variables to be optimized

    # Input Variables:
    # innervertices = the tensor we are trying to optimize, holding the coordinates of the innervertices
    # allvertices = all of the vertices in the Graph
    # innervertexindices = a list that holds the indices of the innervertices tensor within the allvertices array (aka the actual indices of the inner vertices within the graph)
    # aeslist = list where each row corresponds to one inner edge (i,j) and its adjacent faces with format [i, l, j, k]
    # withoutervertices = 
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

def AESallvertexenergy(vertices, aeslist: np.ndarray):
    '''Doubly same as above, but the outer vertices are no longer fixed

    # Input Variables:
    # innervertices = the tensor we are trying to optimize, holding the coordinates of the innervertices
    # allvertices = all of the vertices in the Graph
    # innervertexindices = a list that holds the indices of the innervertices tensor within the allvertices array (aka the actual indices of the inner vertices within the graph)
    # aeslist = list where each row corresponds to one inner edge (i,j) and its adjacent faces with format [i, l, j, k]
    # withoutervertices = 
    '''
    
    i = vertices[:,aeslist[:,0]]
    k = vertices[:,aeslist[:,1]]
    j = vertices[:,aeslist[:,2]]
    l = vertices[:,aeslist[:,3]]

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

def outeredgefixer(vertices, outeredges: np.ndarray, edgelengths):
    '''It is rather difficult to fix the outer edge lengths, which we wish to use as constants to prevent our graphs from collapsing
    This function serves as a punishment on the outer edge lengths, to "softly" prevent the graph collapse without actually fixing the outer edges

    # Input Variables (let oe be the number of outer edges):
    # vertices = 2 x n array holding the vertex positions of our Graph, probably as a tensor to be optimized
    # outeredges = 2 x oe array holding the index pairs that describe each outer edge (i.e. the edge between vertex 1 and 2 shows up as [1,2])
    # edgelengths = column vector of size oe holding the lengths we want the outer edges to be. 

    # Output:
    # energy that represents how different the outeredge lengths are from their "goal lengths"
    '''

    edgevectors = vertices[:,outeredges[1,:]] - vertices[:,outeredges[0,:]]
    curedgelengths = torch.linalg.norm(edgevectors, axis = 0)

    return torch.sum((edgelengths - curedgelengths)**2)
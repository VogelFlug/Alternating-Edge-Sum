# This file holds all the different energy functions used throughout different files
import numpy as np
import torch

import sys
sys.path.insert(0, 'C:/Users/ich/Desktop/Uni/Alternating-Edge-Sum')

import util.Graphutil
from util.TwoDGraph import TwoDGraph

def SnapshotAES(Graph: TwoDGraph):
    '''Calculates the AES energy for a given graph, Might be scuffed since implementation is copied from regular AESenergy. 
    This had a TODO: fix statement on it that i never removed, so there might be something wrong with it
    '''

    ie = util.Graphutil.getinneredges(Graph.edgecounter)

    aeslist = util.Graphutil.getAESList(Graph, ie)
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
    # aeslist = list where each row corresponds to one inner edge (i,j) and its adjacent faces with format [i, k, j, l]
    # withoutervertices = TODO, summarize this function and the next
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
    return energy

def AESallvertexenergy(vertices, aeslist: np.ndarray):
    '''Doubly same as above, but the outer vertices are no longer fixed

    # Input Variables:
    # innervertices = the tensor we are trying to optimize, holding the coordinates of the innervertices
    # allvertices = all of the vertices in the Graph
    # innervertexindices = a list that holds the indices of the innervertices tensor within the allvertices array (aka the actual indices of the inner vertices within the graph)
    # aeslist = list where each row corresponds to one inner edge (i,j) and its adjacent faces with format [i, k, j, l]
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
    return energy


def AESedgeenergy(edges, aeslist: np.ndarray):
    '''AES energy but no longer reliant on vertices. Instead we are going for the edgelengths directly

    # Input Variables:
    # edges: Edgevariable (probably tensor) on which this runs 
    # aeslist = list where each row corresponds to one inner edge (i,j) and its adjacent faces with format [i, k, j, l]

    TODO: not sure how to implement this. Gotta figure out how to avoid confusion on the matrix, whether [i,j] or [j,i] gets changed. 
    Current idea: [i,j] and i is always the smaller one 
    '''
    ik = torch.abs(edges[np.minimum(aeslist[:,0],aeslist[:,1]),np.maximum(aeslist[:,0],aeslist[:,1])])
    kj = torch.abs(edges[np.minimum(aeslist[:,1],aeslist[:,2]),np.maximum(aeslist[:,1],aeslist[:,2])])
    jl = torch.abs(edges[np.minimum(aeslist[:,2],aeslist[:,3]),np.maximum(aeslist[:,2],aeslist[:,3])])
    li = torch.abs(edges[np.minimum(aeslist[:,3],aeslist[:,0]),np.maximum(aeslist[:,3],aeslist[:,0])])

    #energy for each inner edge = (|ik| - |kj| + |jl| - |li|) ^ 2
    energiez = (ik - kj + jl - li)
    
    energy = torch.sum(energiez ** 2)

    # graphy = torchviz.make_dot(energy, params={v: k for v, k in enumerate(list(innervertices))})
    # graphy.render("AEStree", format="png")
    return energy


def edgefixer(curredgelengths, goallengths):
    '''It is rather difficult to fix the outer edge lengths, which we wish to use as constants to prevent our graphs from collapsing
    This function serves as a punishment on the outer edge lengths, to "softly" prevent the graph collapse without actually fixing the outer edges
    We also use the function to reconstruct a graph from the appropriate edgelengths, cause fuck it we ball

    # Input Variables:
    # curedgelengths = our current edge lengths
    # edgelengths = column vector holding the lengths we want the edges to be. 

    # Output:
    # energy that represents how different the edge lengths are from their "goal lengths"
    '''
    return torch.sum((goallengths - curredgelengths)**2)

def anglesum(Surrounds, edgetensor, vertexnr = 0):
    '''We want the sum of angles around all inner vertices to be 360 degrees. For this, we use the law of cosines on each face surrounding the vertex to check its angle.

    # Input Variables:
    # Surrounds: A list of list of lists holding the surroundings of each vertex (read Graphutil.getsurroundingedgelist for clearer explanation)
    # edgetensor: The edgetensor we wish to optimize

    # Output:
    # A vector holding the sum of all surrounding angles for vertex i at index i. TODO: find out if thats in rad or deg
    '''
    invertexnr = len(Surrounds)
    
    anglesums = np.zeros(invertexnr)
    # Now for the outer loop: this just runs once per vertex (i.e. per list in surrounds) and gets the anglesum for that vertex
    for i in range(invertexnr):
        faces = Surrounds[i]
        # this is where the magic happens, we simply calculate angle at the vertex i for that face and add it to its anglesum
        for [a,b,c] in faces:
            temp = a**2 + b**2 - c**2
            temp2 = temp/(2*a*b)
            gamma = np.arccos(temp2)
            anglesums[i] += gamma
            
    return anglesums









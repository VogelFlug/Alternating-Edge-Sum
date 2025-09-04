import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from .TwoDGraph import TwoDGraph

#outer edges only appear once
def getouteredges(edgecounter):
    a, b = np.where(edgecounter == 1)
    list = zip(a,b)
    return {tuple(sorted(pair)) for pair in list}

#inner edges appear twice
def getinneredges(edgecounter) -> set[tuple[int,int]]:
    a, b = np.where(edgecounter == 2)
    list = zip(a,b)
    return {tuple(sorted(pair)) for pair in list}
    
def getoutervertices(edgelist):
    vertices = set()
    for i,j in edgelist:
        vertices.add(i)
        vertices.add(j)
    return tuple(vertices)

def getinnervertices(vertexnumber, outvertices):
    return tuple(set(range(0,vertexnumber)).difference(outvertices))

#This plots a graph in the provided plot
def showGraph(Graph: TwoDGraph, fullplot):
    fullplot.scatter(Graph.vertices[0,:], Graph.vertices[1,:], color = "red")
    #write index to make sure no swapping around happens
    for idx in range(Graph.vertices.shape[1]):
        x, y = Graph.vertices[0, idx], Graph.vertices[1, idx]
        fullplot.text(x, y, str(idx), fontsize=9, color="blue")


    faces : tuple[int,int,int] = Graph.faces 
    vertices = Graph.vertices
    for [i,j,k] in faces: # type: ignore
        # Get determinant to check for valid orientation
        reali, realj, realk = vertices[:,i],vertices[:,j],vertices[:,k]
        area = np.linalg.det([realj-reali, realk-reali])
        if(area >= 0):
            t = Polygon([reali,realj,realk], color = "green")
            fullplot.add_patch(t)
            fullplot.plot([reali[0],realj[0]], [reali[1],realj[1]], color = "black")
            fullplot.plot([realj[0],realk[0]], [realj[1],realk[1]], color = "black")
            fullplot.plot([reali[0],realk[0]], [reali[1],realk[1]], color = "black")
        else:
            t = Polygon([reali,realj,realk], color = "red")
            fullplot.add_patch(t)
            fullplot.plot([reali[0],realj[0]], [reali[1],realj[1]], color = "black")
            fullplot.plot([realj[0],realk[0]], [realj[1],realk[1]], color = "black")
            fullplot.plot([reali[0],realk[0]], [reali[1],realk[1]], color = "black")

    fullplot.axis("equal")

#AESlist is a list where each row corresponds to one inner edge and its adjacent faces with format [i, l, j, k]
def getAESList(Graph: TwoDGraph, inneredges: set[tuple[int,int]]) -> np.ndarray:
    triangelist = np.zeros((len(inneredges),4))
    for count, [i, j] in enumerate(inneredges):
        triangelist[count,0] = i
        triangelist[count,2] = j
        neighbours = list(Graph.neighbourhood[i].intersection(Graph.neighbourhood[j]))
        triangelist[count,1] = neighbours[0]
        triangelist[count,3] = neighbours[1]

    return triangelist.astype(int)

#Tutteembedding directly via Laplace Matrix
def standardtuttembedding(Graph: TwoDGraph):
    oe = getouteredges(Graph.edgecounter)
    ie = getinneredges(Graph.edgecounter)

    ov = getoutervertices(oe)
    ocount = len(ov)
    iv = getinnervertices(Graph.vertexnumber, ov)
    icount = len(iv)

    #First get matrix L for inner vertex positions. The diagonal is filled with 1 to represent the vertex itself and the rest is just subtracting the inner vertex if it is a neighbour of the one of interest (divided by the degree cause Tutte)
    Lx = np.zeros((icount, icount))
    Ly = np.zeros((icount, icount))
    #This is just so we can remember the degree for later when we use it with the right side 
    degreematrix = np.zeros((icount, icount))

    for i in range(0,icount):
        nh = Graph.neighbourhood[iv[i]]
        nhnr = len(nh)
        Lx[i,i] = Ly[i,i] = 1
        degreematrix[i,i] = 1/nhnr

        innerneighbours = tuple(nh.intersection(iv))
        for neighbour in innerneighbours:
            index = iv.index(neighbour)
            Lx[i,index] = Ly[i,index] = -1/nhnr
    
    #print(Lx)

    #Now for the right side of the equation. Basically works the same as L, but with the outer vertices instead.
    outx = Graph.vertices[0,ov[:]]
    outy = Graph.vertices[1,ov[:]]
    Bx = By = np.zeros((icount,ocount))
    for i in range(0, ocount):
        innerneighbours = tuple(Graph.neighbourhood[ov[i]].intersection(iv))
        for neighbour in innerneighbours:
            index = iv.index(neighbour)
            Bx[index,i] = By[index,i] = 1
    
    Bxvec = np.matmul(degreematrix,np.matmul(Bx, outx))
    Byvec = np.matmul(degreematrix,np.matmul(By, outy))
    
    #This code didnt work until i put in this print statement?
    #print(Bxvec)

    #solve Tutte linear system of equations
    innervertices = np.zeros((2, icount))
    innervertices[0,:] = np.linalg.solve(Lx, Bxvec)
    innervertices[1,:] = np.linalg.solve(Ly, Byvec)

    #insert new vertex positions into graph
    newvertices = Graph.vertices.copy()
    for counter, i in enumerate(iv):
        newvertices[:,i] = innervertices[:,counter]

    #create new Graph from vertices
    newGraph = TwoDGraph(vertices=newvertices, faces = Graph.faces)

    return newGraph

#Calculates the AES energy for a given graph, Might be scuffed since implementation is copied from regular AESenergy, TODO: fix
def SnapshotAES(Graph: TwoDGraph):
    ie = getinneredges(Graph.edgecounter)

    aeslist = getAESList(Graph, ie)
    fullvertex = Graph.vertices
    
    i = fullvertex[:,aeslist[:,0]]
    k = fullvertex[:,aeslist[:,1]]
    j = fullvertex[:,aeslist[:,2]]
    l = fullvertex[:,aeslist[:,3]]

    #energy for each inner edge = (|ik| - |kj| + |jl| - |li|) ^ 2
    energies = (np.linalg.norm(k-i, axis=0) - np.linalg.norm(j-k, axis=0) + np.linalg.norm(l-j, axis=0) - np.linalg.norm(i-l, axis=0))
    
    energy = np.sum(energies ** 2)

    return np.sum(energies ** 2)
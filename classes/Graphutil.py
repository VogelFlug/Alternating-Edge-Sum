import numpy as np
from matplotlib import pyplot as plt
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
    return tuple(set(range(1,vertexnumber)).difference(outvertices))

def showGraph(Graph: TwoDGraph):
    plt.scatter(Graph.vertices[0,:], Graph.vertices[1,:], color = "red")
    faces : tuple[int,int,int] = Graph.faces 
    vertices = Graph.vertices
    for [i,j,k] in faces:
        plt.plot([vertices[0,i],vertices[0,j]],[vertices[1,i],vertices[1,j]])
        plt.plot([vertices[0,j],vertices[0,k]],[vertices[1,j],vertices[1,k]])
        plt.plot([vertices[0,i],vertices[0,k]],[vertices[1,i],vertices[1,k]])
    plt.show()

def getAESList(Graph: TwoDGraph, inneredges: set[tuple[int,int]]):
    triangelist = np.zeros((len(inneredges),4))
    for count, [i, j] in enumerate(inneredges):
        triangelist[count,0] = i
        triangelist[count,2] = j
        neighbours = list(Graph.neighbourhood[i].intersection(Graph.neighbourhood[j]))
        triangelist[count,1] = neighbours[0]
        triangelist[count,3] = neighbours[1]

    return triangelist

#Tutteembedding directly via Laplace Matrix
def standardtuttembedding(Graph: TwoDGraph):
    oe = getouteredges(Graph.edgecounter)
    ie = getinneredges(Graph.edgecounter)

    ov = getoutervertices(oe)
    ocount = len(ov)
    iv = getinnervertices(Graph.vertexnumber, ov)
    icount = len(iv)

    #First get matrix L for inner vertex positions. The diagonal is filled with the degree of the inner vertex it represents, L_{i,j} is -1 if (i,j) is an inneredge
    Lx = np.zeros((icount, icount))
    Ly = np.zeros((icount, icount))
    for i in range(0,icount):
        nh = Graph.neighbourhood[iv[i]]
        nhnr = len(nh)
        Lx[i,i] = Ly[i,i] = nhnr

        innerneighbours = tuple(nh.intersection(iv))
        for neighbour in innerneighbours:
            index = iv.index(neighbour)
            Lx[i,index] = Ly[i,index] = -1
    
    print(Lx)

    #Now for the right side of the equation:        
    outx = Graph.vertices[0,ov[:]]
    outy = Graph.vertices[1,ov[:]]
    Bx = By = np.zeros((ocount,icount))
    for i in range(0, ocount):
        innerneighbours = tuple(Graph.neighbourhood[ov[i]].intersection(iv))
        for neighbour in innerneighbours:
            index = iv.index(neighbour)
            Bx[index,i] = By[index,i] = 1
    
    Bxvec = np.matmul(Bx, outx)
    Byvec = np.matmul(By, outy)
    
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

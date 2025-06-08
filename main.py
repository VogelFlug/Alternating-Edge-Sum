#Imports
import numpy as np
import torch
import sys
from io import StringIO

#Internal Imports
from classes.TwoDGraph import TwoDGraph
import classes.Graphutil as util

def standardtuttembedding(Graph: TwoDGraph):
    oe = util.getouteredges(Graph.edgecounter)
    ie = util.getinneredges(Graph.edgecounter)

    ov = util.getoutervertices(oe)
    ocount = len(ov)
    iv = util.getinnervertices(Graph.vertexnumber, ov)
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
            

    #Now for the right side of the equation:        
    outx = Graph.vertices[0,ov[:]]
    outy = Graph.vertices[1,ov[:]]
    Bx = By = np.zeros((ocount,icount))
    for i in range(0, ocount):
        innerneighbours = tuple(Graph.neighbourhood[ov[i]].intersection(iv))
        for neighbour in innerneighbours:
            index = iv.index(neighbour)
            Bx[i,index] = By[i,index] = 1
    
    Bxvec = np.matmul(Bx, outx)
    Byvec = np.matmul(By, outy)
    
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

def main(autopath: str):
    # Get datafile
    x = autopath
    print(x)
    data = ""
    with open(x , "r") as f:
        data = f.read()
    Graph = TwoDGraph(vgl = data)
    # print(Graph.vertices)
    # print(Graph.faces)
    # print(Graph.neighbourhood)
    # print(Graph.edgecounter)
    # oe = util.getouteredges(Graph.edgecounter)
    # ie = util.getinneredges(Graph.edgecounter)
    # print(oe)
    # print(ie)
    TutteGraph = standardtuttembedding(Graph)
    util.showGraph(Graph)
    print(TutteGraph.vertices)
    util.showGraph(TutteGraph)

if __name__ == '__main__':
    main(sys.argv[1])



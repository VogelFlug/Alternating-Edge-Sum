import numpy as np
from matplotlib import pyplot as plt
from . import TwoDGraph

#outer edges only appear once
def getouteredges(edgecounter):
    a, b = np.where(edgecounter == 1)
    list = zip(a,b)
    return {tuple(sorted(pair)) for pair in list}

#inner edges appear twice
def getinneredges(edgecounter):
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
    faces = Graph.faces
    vertices = Graph.vertices
    for i,j,k in faces:
        plt.plot([vertices[0,i],vertices[0,j]],[vertices[1,i],vertices[1,j]])
        plt.plot([vertices[0,j],vertices[0,k]],[vertices[1,j],vertices[1,k]])
        plt.plot([vertices[0,i],vertices[0,k]],[vertices[1,i],vertices[1,k]])
    plt.show()
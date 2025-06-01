import numpy as np
#from . import TwoDGraph

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
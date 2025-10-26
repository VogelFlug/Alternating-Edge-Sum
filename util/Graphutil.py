import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import copy

import sys
sys.path.insert(0, 'C:/Users/ich/Desktop/Uni/Alternating-Edge-Sum')
sys.path.insert(0, 'C:/Users/daveb/OneDrive/Desktop/Uni/Alternating-Edge-Sum/')

from util.TwoDGraph import TwoDGraph

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

def getalledges(edgecounter):
    a, b = np.where(np.triu(edgecounter) != 0)
    return np.array([a,b]).T
    
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


def newreconstructfromedgelengths(faces, edgelengths, dimensions = 2):
    '''Reconstructs a Graph from edgelengths by recreating the triangles one by one. We create the first face manually and from there attach the surrounding ones one by one
    
    # Input variables:
    # faces = a list of all the faces of the Graph, of type list[int,int,int]
    # edgelengths = edgelengths we have generated, will be in the form of a matrix where (i,j) needs to be the edgelength between i and j (otherwise zero) for all i,j. if i had to be smaller than j, this would just be painful
    # dimensions = just in case i wanna keep this function for 3 dimensions

    # Output: New Graph with those edgelengths

    '''
    facequeue = copy.deepcopy(faces)
    # we keep track of the vertices we have already determined#
    vertextracker = []
    nr_vertices = edgelengths.shape[0]
    newvertices = np.zeros((dimensions,nr_vertices))

    # Hardforce first face. 
    firstface = facequeue[0]
    facequeue.remove(firstface)
    i, j, k = firstface[0], firstface[1], firstface[2]
    ij = edgelengths[i,j]
    ik = edgelengths[i,k]
    jk = edgelengths[j,k]

    # the first vertex gets put into the origin for simplicity's sake, the second one goes to the right of the first one. This will probably cause a big roto-translation of the original graph, but I don't exactly care
    newvertices[:,i] = [0,0]
    newvertices[:,j] = [ij,0]

    # This is the hard part and the part we repeat in the loop: getting the third vertex position. The math for this has been done beforehand
    # Its also where we split off between dimensionalities, as the first two vertices work just as well in 2 dimensions
    if (dimensions == 2):
        # get the point on IJ that is orthogonal to K
        ratio = ((ik ** 2) - (jk ** 2) + (ij ** 2)) / (2 * ij)
        p = newvertices[:,i] + ratio/ij * (newvertices[:,j] - newvertices[:,i])
        height = np.sqrt((ik ** 2) - (ratio ** 2))

        # now we get K from P
        x_k1, x_k2 = p[0] + height * ((newvertices[1,j] - newvertices[1,i]) / ij), p[0] - height * ((newvertices[1,j] - newvertices[1,i]) / ij)
        y_k1, y_k2 = p[1] - height * ((newvertices[0,j] - newvertices[0,i]) / ij), p[1] + height * ((newvertices[0,j] - newvertices[0,i]) / ij)

        # Lastly we check which point we want. One of these should produce a positive determinante
        point1 = np.array([x_k1, y_k1])
        determinante1 = np.linalg.det([newvertices[:,j] - newvertices[:,i], point1 - newvertices[:,i]])
        if(determinante1 > 0):
            newvertices[:,k] = point1
        else:
            newvertices[:,k] = np.array([x_k2, y_k2])

        #we also add all the vertices to the tracker
        vertextracker.append(i), vertextracker.append(j), vertextracker.append(k) # type: ignore

        # Now we basically repeat what we did for k in the last step for all of the faces. If a face shares two vertices with ones we have already, we can work on that. Otherwise it is skipped for now
        while(len(vertextracker) < nr_vertices):
            for face in facequeue:
                crossover = set(face).intersection(vertextracker)
                if(len((set(face).intersection(vertextracker))) == 2): 
                    facequeue.remove(face)
                    k = (set(face) - crossover).pop()
                    index = face.index(k)
                    i, j = face[(1 + index) % 3], face[(2 + index) % 3]
                    
                    ij = edgelengths[i,j]
                    ik = edgelengths[i,k]
                    jk = edgelengths[j,k]

                    # now we literally copy what we did above
                    ratio = ((ik ** 2) - (jk ** 2) + (ij ** 2)) / (2 * ij)
                    p = newvertices[:,i] + (ratio/ij * (newvertices[:,j] - newvertices[:,i]))
                    height = np.sqrt((ik ** 2) - (ratio ** 2))
                    x_k1, x_k2 = p[0] + height * ((newvertices[1,j] - newvertices[1,i]) / ij), p[0] - height * ((newvertices[1,j] - newvertices[1,i]) / ij)
                    y_k1, y_k2 = p[1] - height * ((newvertices[0,j] - newvertices[0,i]) / ij), p[1] + height * ((newvertices[0,j] - newvertices[0,i]) / ij)

                    point1 = np.array([x_k1, y_k1])
                    determinante1 = np.linalg.det([newvertices[:,j] - newvertices[:,i], point1 - newvertices[:,i]])
                    if(determinante1 > 0):
                        newvertices[:,k] = point1
                    else:
                        newvertices[:,k] = np.array([x_k2, y_k2])

                    vertextracker.append(k)

    
    newGraph = TwoDGraph(vertices=newvertices, faces=faces)
    return newGraph


def spherepacker(Graph: TwoDGraph, edges, edgelengths, dimensions = 2):
    '''Given a graph G = (V,E) where sphere packing is possible (we check for the AES being zero to verify), this function serves to calculate the radius of the sphere for each vertex that would make this packing possible.
    The Matrix simply adds the two respective radii to sum up to the edgelength.
    TODO: Furhter implement a visualization of the spheres

    # Input Variables: 
    # edges = E x 2 array, with all the edges in form (i,j) where i and j are the connected vertices
    # edgelengths = edgelengths we have generated, will be in the form of a matrix where (i,j) needs to be the edgelength between i and j (otherwise zero) for all i,j. if i had to be smaller than j, this would just be painful
    # dimensions = Currently only supports 2 dimensions TODO: implement 3

    # Output:
    # array of size |V|, that gives the radius for a vertex at the respective index
    '''
    vertices = Graph.vertices
    edgenr = edges.shape[0]

    # The connectivity matrix has one row per edge and that row is entirely 0 except for the two vertices that make that edge
    # The edgevector just holds the respective edge length
    connectivity = np.zeros((edgenr,vertices.shape[1]))
    edgevector = np.zeros((edgenr,1))
    for i in range(edgenr):
        connectivity[i, edges[i,0]] = connectivity[i, edges[i,1]] = 1
        edgevector[i] = edgelengths[edges[i,0], edges[i,1]]

    radii = np.squeeze(np.linalg.lstsq(connectivity, edgevector)[0])
    return radii

def visualizecircles(vertices, radii, subplot):
    '''Draw the circle with given radius at the position of the vertex of the same index
    '''
    for i in range(radii.shape[0]):
        circle = plt.Circle(vertices[:,i], radii[i]) # type: ignore
        subplot.add_patch(circle)

    subplot.axis('equal')











'''for testing purposes'''
# filepath = "data/2dfolder/fulldata/testfile.txt"
# # #edges = np.array([[0,1,1,0, 0.707106781186],[1, 0, 0, 1, 0.707106781186],[1,0,0,1, 0.707106781186],[0,1,1,0, 0.707106781186],[0.707106781186,0.707106781186,0.707106781186,0.707106781186,0]])



# data = ""
# with open(filepath , "r") as f:
#     data = f.read()
# Graph = TwoDGraph(vgl = data)
# a, b = np.where(Graph.edgecounter != 0)
# edges = np.array([a,b]).T
# edgelengths = np.zeros((Graph.vertices.shape[1],Graph.vertices.shape[1]))
# for i,j in edges:
#     edgelengths[i,j] = edgelengths[j,i] = np.linalg.norm(Graph.vertices[:,i] - Graph.vertices[:,j])

# # print(getalledges(Graph.edgecounter))

# # radii = spherepacker(Graph, edges, edgelengths)
# # print(radii)

# fig, axs = plt.subplots()
# showGraph(Graph, axs)
# plt.show()

# newGraph = newreconstructfromedgelengths(list(Graph.faces), edgelengths)


# fig, axs = plt.subplots()
# showGraph(newGraph, axs)
# plt.show()
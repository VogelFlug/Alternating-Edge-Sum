import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import copy

import sys
sys.path.insert(0, 'C:/Users/ich/Desktop/Uni/Alternating-Edge-Sum')

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


def reconstructfromedgelengths(Graph: TwoDGraph, edgelengths, learnrate = 0.001):
    '''Reconstructs a Graph from edgelengths via optimization.
    
    # Input variables:
    # Graph = Graph with which to start the optimization, will probably just be the original graph.
    # edgelengths = edgelengths we have generated, will be in the form of a matrix where (i,j) needs to be the edgelength between i and j (otherwise zero) for all i<j

    # Output: New Graph determined to be "close enough" to the goal edgelengths

    TODO: why does this struggle to reconstruct some of the graphs?
    '''
    vertices = Graph.vertices
    oe = np.array(list(getouteredges(Graph.edgecounter)))
    ie = np.array(list(getinneredges(Graph.edgecounter)))


    Verttensor = torch.tensor(vertices, requires_grad=True)
    edges = np.concatenate((oe,ie))
    #make the edges sorted so vertex one has lower index than vertex two. This is to make up for my scuffed implementation earlier
    edges[:,0], edges[:,1] = np.minimum(edges[:,0],edges[:,1]), np.maximum(edges[:,0],edges[:,1])
    #to remember the energies so we can plot them afterwards
    energies = []
    # Get the goallengths outside of the loop cause they stay consistent
    goalleng = torch.tensor(edgelengths[edges[:,0], edges[:,1]], requires_grad=False)

    for i in range(2000):
        # First get a vector holding our current edgelengths:
        curredgeleng = torch.linalg.norm(Verttensor[:,edges[:,0]]-Verttensor[:,edges[:, 1]], dim=0)


        # get the energy
        energy = torch.sum((goalleng - curredgeleng)**2)
        energies.append(energy.item())
        
        # get gradient through backpropagation
        energy.backward()
        

        with torch.no_grad():
            Verttensor -= learnrate * Verttensor.grad # type: ignore

        # reset Gradient
        Verttensor.grad.zero_() # type: ignore
        
    vertexs = Verttensor.detach().numpy()
    newGraph = TwoDGraph(vertices=vertexs, faces=Graph.faces)
    return newGraph, np.array(energies)

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
    print(firstface)
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
        vertextracker.append(i), vertextracker.append(j), vertextracker.append(k)

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


'''for testing purposes'''
# filepath = "data/2dfolder/fulldata/megabasicoff.txt"
# edges = np.array([[0,1,1,0, 0.707106781186],[1, 0, 0, 1, 0.707106781186],[1,0,0,1, 0.707106781186],[0,1,1,0, 0.707106781186],[0.707106781186,0.707106781186,0.707106781186,0.707106781186,0]])

# data = ""
# with open(filepath , "r") as f:
#     data = f.read()
# Graph = TwoDGraph(vgl = data)

# newGraph = newreconstructfromedgelengths(list(Graph.faces), edges)


# fig, axs = plt.subplots()
# showGraph(newGraph, axs)
# plt.show()
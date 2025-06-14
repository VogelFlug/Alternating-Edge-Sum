#Imports
import numpy as np
import torch
import torchviz
import sys
from io import StringIO

#Internal Imports
from classes.TwoDGraph import TwoDGraph
import classes.Graphutil as util


#energy function as mean of neighbourpositions squared. Applicable to both x and y
def tutteenergy(innervertices, allvertices: np.ndarray, innervertexindices, neighbourhood: tuple[set[int]], axis):
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

        
            

#Tutteembedding via Gradient Descent
def gradienttutte(Graph: TwoDGraph, learnrate: float):
    vertices = Graph.vertices
    oe = util.getouteredges(Graph.edgecounter)
    ie = util.getinneredges(Graph.edgecounter)

    ov = util.getoutervertices(oe)
    iv = util.getinnervertices(Graph.vertexnumber, ov)

    Xtensor = torch.tensor((vertices[:,iv])[0,:].tolist(), requires_grad=True)
    Ytensor = torch.tensor((vertices[:,iv])[1,:].tolist(), requires_grad=True)

    for i in range(10000):
        #calc energy
        xenergy = tutteenergy(Xtensor, vertices, iv, Graph.neighbourhood, 0)
        yenergy = tutteenergy(Ytensor, vertices, iv, Graph.neighbourhood, 1)
        
        #get gradient through backpropagation
        xenergy.backward() # type: ignore
        yenergy.backward() # type: ignore
        #print(Xtensor)
        #print(Xtensor.grad)
        #print(Ytensor.grad)
        with torch.no_grad():
            Xtensor -= learnrate * Xtensor.grad # type: ignore
            Ytensor -= learnrate * Ytensor.grad # type: ignore

        #print(Xtensor)
        #Gradienten zurücksetzen
        Xtensor.grad.zero_() # type: ignore
        Ytensor.grad.zero_() # type: ignore
    
    vertexs = np.zeros((2,Graph.vertexnumber))
    vertx = Xtensor.detach().numpy()
    verty = Ytensor.detach().numpy()
    for i in range(Graph.vertexnumber):
        if(i in iv):
            vertexs[0,i] = vertx[iv.index(i)]
            vertexs[1,i] = verty[iv.index(i)]
        else:
            vertexs[:,i] = Graph.vertices[:,i]

    newGraph = TwoDGraph(vertices=vertexs, faces=Graph.faces)
    return newGraph


def gradientAES(Graph: TwoDGraph, learnrate: float):
    vertices = Graph.vertices
    oe = util.getouteredges(Graph.edgecounter)
    ie = util.getinneredges(Graph.edgecounter)

    ov = util.getoutervertices(oe)
    iv = util.getinnervertices(Graph.vertexnumber, ov)

    Xtensor = torch.tensor((vertices[:,iv])[0,:].tolist(), requires_grad=True)
    Ytensor = torch.tensor((vertices[:,iv])[1,:].tolist(), requires_grad=True)

    for i in range(10000):
        #calc energy
        xenergy = tutteenergy(Xtensor, vertices, iv, Graph.neighbourhood, 0)
        yenergy = tutteenergy(Ytensor, vertices, iv, Graph.neighbourhood, 1)
        
        #get gradient through backpropagation
        xenergy.backward() # type: ignore
        yenergy.backward() # type: ignore
        #print(Xtensor)
        #print(Xtensor.grad)
        #print(Ytensor.grad)
        with torch.no_grad():
            Xtensor -= learnrate * Xtensor.grad # type: ignore
            Ytensor -= learnrate * Ytensor.grad # type: ignore

        #print(Xtensor)
        #Gradienten zurücksetzen
        Xtensor.grad.zero_() # type: ignore
        Ytensor.grad.zero_() # type: ignore
    
    vertexs = np.zeros((2,Graph.vertexnumber))
    vertx = Xtensor.detach().numpy()
    verty = Ytensor.detach().numpy()
    for i in range(Graph.vertexnumber):
        if(i in iv):
            vertexs[0,i] = vertx[iv.index(i)]
            vertexs[1,i] = verty[iv.index(i)]
        else:
            vertexs[:,i] = Graph.vertices[:,i]

    newGraph = TwoDGraph(vertices=vertexs, faces=Graph.faces)
    return newGraph



#Tutteembedding directly via Laplace Matrix
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




def main(autopath: str):
    # Get datafile
    x = autopath
    #print(x)
    data = ""
    with open(x , "r") as f:
        data = f.read()
    Graph = TwoDGraph(vgl = data)
    util.showGraph(Graph)


    #TutteGraph = standardtuttembedding(Graph)
    #util.showGraph(Graph)
    #print(Graph.vertices.T)
    # print("\n\n\n")
    #print(TutteGraph.vertices.T)
    #util.showGraph(TutteGraph)

    #FakeTuttegraph = gradienttutte(Graph, 0.001)
    #util.showGraph(FakeTuttegraph)
    print(util.getAESList(Graph, util.getinneredges(Graph.edgecounter)))

if __name__ == '__main__':
    main(sys.argv[1])



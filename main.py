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

    Xtensor = torch.tensor((vertices[0,iv]).tolist(), requires_grad=True)
    Ytensor = torch.tensor((vertices[1,iv]).tolist(), requires_grad=True)

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

def AESenergy(innervertices, allvertices: np.ndarray, innervertexindices, aeslist: np.ndarray):
    fullvertex = torch.tensor(allvertices.tolist(), requires_grad=False)
    #This combines our fixed outside vertices with the changing inner vertices
    for count, i in enumerate(innervertexindices):
        fullvertex[:,i] = innervertices[:,count]
    
    i = fullvertex[:,aeslist[:,0]]
    k = fullvertex[:,aeslist[:,1]]
    j = fullvertex[:,aeslist[:,2]]
    l = fullvertex[:,aeslist[:,3]]
    # print(aeslist)
    # print(i)
    # print(j)
    # print(k)
    # print(l)

    #energy for each inner edge = (|ik| - |kj| + |jl| - |li|) ^ 2
    energies = (torch.linalg.norm(i-k, dim=0) - torch.linalg.norm(j-k, dim=0) + torch.linalg.norm(l-j, dim=0) - torch.linalg.norm(i-l, dim=0))
    return torch.sum(energies ** 2)

def gradientAES(Graph: TwoDGraph, learnrate: float):
    vertices = Graph.vertices
    oe = util.getouteredges(Graph.edgecounter)
    ie = util.getinneredges(Graph.edgecounter)

    ov = util.getoutervertices(oe)
    iv = util.getinnervertices(Graph.vertexnumber, ov)

    Verttensor = torch.tensor((vertices[:,iv]).tolist(), requires_grad=True)

    AESlist = util.getAESList(Graph, ie)

    for i in range(5000):
        #calc energy
        energy = AESenergy(Verttensor, vertices, iv, AESlist)
        #print(energy)
        
        #get gradient through backpropagation
        energy.backward()
        

        with torch.no_grad():
            Verttensor -= learnrate * Verttensor.grad # type: ignore

        #print(Xtensor)
        #Gradienten zurücksetzen
        Verttensor.grad.zero_() # type: ignore
    
    vertexs = np.zeros((2,Graph.vertexnumber))
    gradientfinal = Verttensor.detach().numpy()
    for i in range(Graph.vertexnumber):
        if(i in iv):
            vertexs[:,i] = gradientfinal[:,iv.index(i)]
        else:
            vertexs[:,i] = Graph.vertices[:,i]

    newGraph = TwoDGraph(vertices=vertexs, faces=Graph.faces)
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

    AESgraph = gradientAES(Graph, 0.001)
    print(AESgraph.vertices)
    util.showGraph(AESgraph)

if __name__ == '__main__':
    main(sys.argv[1])



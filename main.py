#Imports
import numpy as np
import torch
#import torchviz
from matplotlib import pyplot as plt
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

    for i in range(2000):
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

    #energy for each inner edge = (|ik| - |kj| + |jl| - |li|) ^ 2
    energiez = (torch.linalg.norm(k-i, dim=0) - torch.linalg.norm(j-k, dim=0) + torch.linalg.norm(l-j, dim=0) - torch.linalg.norm(i-l, dim=0))
    
    energy = torch.sum(energiez ** 2)

    # graphy = torchviz.make_dot(energy, params={v: k for v, k in enumerate(list(innervertices))})
    # graphy.render("AEStree", format="png")
    return torch.sum(energiez ** 2)

def gradientAES(Graph: TwoDGraph, learnrate: float, loops: int):
    vertices = Graph.vertices
    oe = util.getouteredges(Graph.edgecounter)
    ie = util.getinneredges(Graph.edgecounter)

    ov = util.getoutervertices(oe)
    iv = util.getinnervertices(Graph.vertexnumber, ov)
    if(len(iv) == 0):
        raise Exception("What is the point without innervertices!?")

    Verttensor = torch.tensor((vertices[:,iv]).tolist(), requires_grad=True)

    AESlist = util.getAESList(Graph, ie)

    #to remember the energies so we can plot them afterwards
    energies = []

    for i in range(loops):
        #calc energy
        energy = AESenergy(Verttensor, vertices, iv, AESlist)
        energies.append(energy.item())
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
    return newGraph, np.array(energies)




def gradientAESfixededges(Graph: TwoDGraph, learnrate: float):
    vertices = Graph.vertices
    oe = util.getouteredges(Graph.edgecounter)
    ie = util.getinneredges(Graph.edgecounter)

    ov = util.getoutervertices(oe)
    iv = util.getinnervertices(Graph.vertexnumber, ov)
    if(len(iv) == 0):
        raise Exception("What is the point without innervertices!?")
    return


def main(Graph: TwoDGraph, outputpath: str, attempts = 1, stepsize = 2000):   
    '''Main function. This takes a 2 dimensional Graph, calculates its Tutte Embedding (And the Tutte-Embeddings AES energy for reference) 
    and then uses Gradient Descent to minimize the AES energy in an attempt to create an Embedding based on sphere packing. The results, including the AES energy over time all end up in a pdf

    # Input Variables:
    # Graph = Input Graph. You're welcome
    # outputpath = Where the PDF ends up
    # attempts = How many times you wanna run this with different numbers of loops. Mostly serves debugging to ease visualize the change in the graph over time.
    # stepsize = How many more training steps each iteration takes then the last. Aka, attempt 1 has stepsize many loops, attempt 2 has 2*stepsize many loops etc.

    # Output:
    # The pdf. Thas it for now
    
    # TODO: Find a way to show the x axes for lots of attempts?
    '''



    #create the plots
    fig, axs = plt.subplots(1 + attempts,2)

    #plot input graph for reference
    axs[0,0].set_title("Original Graph", fontsize = 7)
    util.showGraph(Graph, axs[0,0])

    #plot the Tutte Embedding of the original Graph and add the AES value to the side (only 6 decimal points cause otherwise it will get very messy)
    axs[0,1].set_title("Tutte Embedding", fontsize = 7)
    TutteGraph = util.standardtuttembedding(Graph)
    util.showGraph(TutteGraph, axs[0,1])
    axs[0,1].text(1.1,0.5, "    AES energy\n  for Tutte: \n     " + str(format(util.SnapshotAES(TutteGraph),".8f")), transform=axs[0,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)

    
    for i in range(1, 1 + attempts):
        AESgraph, energies = gradientAES(Graph, 0.001, i * stepsize)

        axs[i,0].set_title("AES minimized graph", fontsize = 7, y = -0.25)
        util.showGraph(AESgraph, axs[i,0])

        axs[i,1].set_title("AES energy over optimization",fontsize = 7, y = -0.25)
        axs[i,1].plot(energies)
        axs[i,1].text(1.1,0.5, " Final AES\n  energy:\n     " + str(format(energies[-1], ".8f")), transform=axs[i,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)

    # TutteAES, Tutteenergies = gradientAES(TutteGraph, 0.001)
    
    # axs[2,0].set_title("AES minimized graph starting with Tutte", fontsize = 7)
    # util.showGraph(TutteAES, axs[2,0])

    # axs[2,1].set_title("AES energy over optimization",fontsize = 7)
    # axs[2,1].plot(Tutteenergies)
    # axs[2,1].text(1.1,0.5, "Final AES energy: " + str(format(Tutteenergies[-1], ".8f")), transform=axs[2,1].transAxes,  rotation = 270, va = "center", ha="center", fontsize=7)

    
    plt.savefig(outputpath + "_results.pdf")

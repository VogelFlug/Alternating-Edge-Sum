#Imports
import numpy as np
import torch
#import torchviz
from matplotlib import pyplot as plt
import sys
from io import StringIO

#Internal Imports
from util.TwoDGraph import TwoDGraph
import util.Graphutil as gutil
import util.optimizationenergies as optimizers



def gradientAESfixedvertices(Graph: TwoDGraph, learnrate: float, loops: int):
    '''Attempts optimization of a Graph through gradient descent of AES energy,
    where the outter vertices are fixed and the inner vertex positions are to be optimized, in an attempt to achieve an embedding of that Graph
    
    # Input Variables:
    # Graph = Base Graph that is to be optimize
    # learnrate = rate at which each gradient descent step is taken
    # loops: The number of gradient descent steps taken in total

    # Output Variables:
    # newGraph = Graph produced by the optimization. Not necessarily an embedding, depends on the base Graph
    # energies = row vector of size loops, holding the energy of the Graph at each step of the optimization. Used for visualization and subsequent debugging
    '''
    vertices = Graph.vertices
    oe = gutil.getouteredges(Graph.edgecounter)
    ie = gutil.getinneredges(Graph.edgecounter)

    ov = gutil.getoutervertices(oe)
    iv = gutil.getinnervertices(Graph.vertexnumber, ov)
    if(len(iv) == 0):
        raise Exception("What is the point without innervertices!?")

    Verttensor = torch.tensor((vertices[:,iv]).tolist(), requires_grad=True)

    AESlist = gutil.getAESList(Graph, ie)

    #to remember the energies so we can plot them afterwards
    energies = []

    for i in range(loops):
        #calc energy
        energy = optimizers.AESvertexenergy(Verttensor, vertices, iv, AESlist)
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



#TODO will be gradientAES with fixedouteredges and using the inner edges as variables to optimize
def gradientAESfixededges(Graph: TwoDGraph, learnrate: float):
    vertices = Graph.vertices
    oe = gutil.getouteredges(Graph.edgecounter)
    ie = gutil.getinneredges(Graph.edgecounter)

    ov = gutil.getoutervertices(oe)
    iv = gutil.getinnervertices(Graph.vertexnumber, ov)
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
    gutil.showGraph(Graph, axs[0,0])

    #plot the Tutte Embedding of the original Graph and add the AES value to the side (only 6 decimal points cause otherwise it will get very messy)
    axs[0,1].set_title("Tutte Embedding", fontsize = 7)
    TutteGraph = gutil.standardtuttembedding(Graph)
    gutil.showGraph(TutteGraph, axs[0,1])
    axs[0,1].text(1.1,0.5, "    AES energy\n  for Tutte: \n     " + str(format(optimizers.SnapshotAES(TutteGraph),".8f")), transform=axs[0,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)

    
    for i in range(1, 1 + attempts):
        AESgraph, energies = gradientAESfixedvertices(Graph, 0.001, i * stepsize)

        axs[i,0].set_title("AES minimized graph", fontsize = 7, y = -0.25)
        gutil.showGraph(AESgraph, axs[i,0])

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

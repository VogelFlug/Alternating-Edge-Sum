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
        raise Exception("This function doesn't work without inner vertices, as there'd be nothing to optimize")

    Verttensor = torch.tensor((vertices[:,iv]).tolist(), requires_grad=True)

    AESlist = gutil.getAESList(Graph, ie)

    #to remember the energies so we can plot them afterwards
    energies = []

    for i in range(loops):
        #calc energy
        energy = optimizers.AESinnervertexenergy(Verttensor, vertices, iv, AESlist)
        energies.append(energy.item())
        
        # get gradient through backpropagation
        energy.backward()  

        with torch.no_grad():
            Verttensor -= learnrate * Verttensor.grad # type: ignore

        # reset Gradient
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



def gradientAESflexibleedges(Graph: TwoDGraph, learnrate: float, loops: int):
    '''Basically the function above but a penalization on the outer edge length rather than fixing the outer vertices themselves
    '''
    vertices = Graph.vertices
    oe = np.array(list(gutil.getouteredges(Graph.edgecounter)))
    ie = gutil.getinneredges(Graph.edgecounter)

    # to penalize the outer edge lengths, we need to know them first
    edgevectors = vertices[:,oe[1,:]] - vertices[:,oe[0,:]]
    edgelengths = torch.tensor(np.linalg.norm(edgevectors, axis = 0), requires_grad = False)

    Verttensor = torch.tensor(vertices, requires_grad=True)

    AESlist = gutil.getAESList(Graph, ie)

    #to remember the energies so we can plot them afterwards
    energies = []

    for i in range(loops):
        # get the current lenghts of our outer edges
        tmpedgevectors = Verttensor[:,oe[1,:]] - Verttensor[:,oe[0,:]]
        curedgelengths = torch.linalg.norm(tmpedgevectors, axis = 0)

        # calculate energy 
        energy = optimizers.edgefixer(curedgelengths, edgelengths) + optimizers.AESallvertexenergy(Verttensor, AESlist)
        energies.append(energy.item())
        
        # get gradient through backpropagation
        energy.backward()
    
        with torch.no_grad():
            Verttensor -= learnrate * Verttensor.grad # type: ignore

        # Reset Gradient
        Verttensor.grad.zero_() # type: ignore
    

    vertexs = Verttensor.detach().numpy()
    
    newGraph = TwoDGraph(vertices=vertexs, faces=Graph.faces)
    return newGraph, np.array(energies)



def gradientAESoptimizeedges(Graph: TwoDGraph, learnrate: float, loops: int):
    '''We now optimize the edgelengths directly, rather than optimizing the vertices first. 
    
    This also introduces a second optimization in Graphutil which attempts to reconstruct the new graph from its edgelengths. Thus a second energy from that function is returned
    For further explanation, read the documentation for util.Graphutil.reconstructfromedgelengths
    TODO: Debug
    '''
    vertices = Graph.vertices
    oe = np.array(list(gutil.getouteredges(Graph.edgecounter)))
    ie = gutil.getinneredges(Graph.edgecounter)

    # to penalize the outer edge lengths, we need to know them first
    alledges = np.concatenate((oe,np.array(list(ie))))
    outeredgevectors = vertices[:,oe[:,1]] - vertices[:,oe[:,0]]
    outeredgelengths = torch.tensor(np.linalg.norm(outeredgevectors, axis = 0), requires_grad = False)

    # rather than a vector we optimize, for purposes of efficiency we introduce a matrix over all vertices that notes the edgelength between
    # vertices i and j in the cell (i,j). Other cells will be left as 0. If this can be done more efficiently than a for loop, that'd be pretty sick
    edgematrix = np.zeros((vertices.shape[1], vertices.shape[1]))
    for edge in alledges:
        edgematrix[edge[0],edge[1]] = edgematrix[edge[1],edge[0]] = np.linalg.norm(vertices[:,edge[1]] - vertices[:,edge[0]])

    edgetensor = torch.tensor(edgematrix, requires_grad = True)

    AESlist = gutil.getAESList(Graph, ie)

    #to remember the energies so we can plot them afterwards
    conditionenergies = []
    energies = []


    for i in range(loops):
        # calc energy 
        # softouteredgeenergy = 10*optimizers.edgefixer(edgetensor[oe[:,0],oe[:,1]], outeredgelengths)
        # conditionenergies.append(softouteredgeenergy.item())

        aesenergy = optimizers.AESedgeenergy(edgetensor, AESlist)
        energies.append(aesenergy.item())
        
        fullenergy = aesenergy #+ softouteredgeenergy
        # get gradient through backpropagation
        fullenergy.backward()

        with torch.no_grad():
            edgetensor.grad[oe[:,0],oe[:,1]] = 0.0 # type: ignore
            edgetensor -= learnrate * edgetensor.grad # type: ignore
            #edgetensor[oe[:,0],oe[:,1]] = outeredgelengths

        torch.abs(edgetensor)

        # Reset Gradient
        edgetensor.grad.zero_() # type: ignore
    edges = edgetensor.detach().numpy()
    edges = np.triu(edges) + np.transpose(np.triu(edges))
    print(edges)

    newGraph = gutil.newreconstructfromedgelengths(list(Graph.faces), edges)
    return newGraph, np.array(energies), np.array(conditionenergies)









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
    
    # TODO: Find a way to adjust the aspect ratio
    '''



    #create the plots
    fig, axs = plt.subplots(1 + attempts,2)

    #plot input graph for reference
    axs[0,0].set_title("Original Graph", fontsize = 7)
    gutil.showGraph(Graph, axs[0,0])

    # plot the Tutte Embedding of the original Graph and add the AES value to the side (only 6 decimal points cause otherwise it will get very messy)
    axs[0,1].set_title("Tutte Embedding", fontsize = 7)
    TutteGraph = gutil.standardtuttembedding(Graph)
    gutil.showGraph(TutteGraph, axs[0,1])
    axs[0,1].text(1.1,0.5, "    AES energy\n  for Tutte: \n     " + str(format(optimizers.SnapshotAES(TutteGraph),".8f")), transform=axs[0,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)

    
    for i in range(1, 1 + attempts):
        AESgraph, energies, outeredenergies = gradientAESoptimizeedges(Graph, 0.01, i * stepsize)

        #axs[i,0].set_title("Soft conditions over optimization",fontsize = 7, y = -0.25)
        #print(outeredenergies[-1])

        axs[i,0].text(1.1,0.5, " Final AES\n  energy:\n     " + str(format(energies[-1], ".8f")), transform=axs[i,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)

        axs[i,0].set_title("AES minimized graph", fontsize = 7, y = -0.25)
        gutil.showGraph(AESgraph, axs[i,0])

        axs[i,1].set_title("outer energy over optimization",fontsize = 7, y = -0.25)
        axs[i,1].plot(energies)
        axs[i,1].text(1.1,0.5, " Final AES\n  energy:\n     " + str(format(energies[-1], ".8f")), transform=axs[i,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)


    # TutteAES, Tutteenergies = gradientAES(TutteGraph, 0.001)
    
    # axs[2,0].set_title("AES minimized graph starting with Tutte", fontsize = 7)
    # util.showGraph(TutteAES, axs[2,0])

    # axs[2,1].set_title("AES energy over optimization",fontsize = 7)
    # axs[2,1].plot(Tutteenergies)
    # axs[2,1].text(1.1,0.5, "Final AES energy: " + str(format(Tutteenergies[-1], ".8f")), transform=axs[2,1].transAxes,  rotation = 270, va = "center", ha="center", fontsize=7)

    
    plt.savefig(outputpath + "_edges_optimized.pdf")

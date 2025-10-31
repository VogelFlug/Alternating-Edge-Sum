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



def gradientAESflexibleedges(Graph: TwoDGraph, loops: int, learnrate = 0.01):
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



def gradientAESoptimizeedges(Graph: TwoDGraph, loops: int, learnrate = 0.01):
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



def optimizeviasvg(Graph: TwoDGraph, loops: int, learnrate = 0.01):
    '''Once again we optimize the edge vectors to keep the problem quadratic. However, through the singular value decomposition of the connectivity matrix (as long as the graph has more edges than vertices),
    we can find a matrix N describing the space to which our optimal solution of edgelengths should be orthogonal.

    TODO 1: Implement cutting of negative Radii (min(0, N^+ @ edges))
    TODO 2: Implement anglesum constraint (pi - anglesum) and find out if this whole thing still functions with an arccos thrown in?
    TODO 3: Implement the Graphing of multiple constraints


    '''
    vertices = Graph.vertices
    edges = gutil.getalledges(Graph.edgecounter)
    vertexnr = vertices.shape[1]
    edgenr = edges.shape[0]
    iv = gutil.ivfromscratch()

    energies = []
    constraintenergies = []

    # The connectivity matrix has one row per edge and that row is entirely 0 except for the two vertices that make that edge
    connectivity = np.zeros((edgenr,vertexnr))
    for i in range(edgenr):
        connectivity[i, edges[i,0]] = connectivity[i, edges[i,1]] = 1

    # Our problem can be broken down to connectivity @ radii = edges. Through singular value decomposition, we can find the core of the connectivity matrix, hidden in left singular vectors. 
    LS, Sig, RS = np.linalg.svd(connectivity)
    N = torch.tensor(LS[:,vertexnr:].T, requires_grad = False)
    # print("right vectors: " , RS),print("Singular values: ", Sig),print("Full lefties:", LS), print("only lefties we want:", LS[:,vertexnr:])

    # And now we get to the actual optimization. Since N is orthogonal to some fitting edgelengths in the question of connectivity @ radii = edges, we wish to achieve exactly that:
    edgetensor = torch.tensor(np.linalg.norm(vertices[:,edges[:,0]] - vertices[:,edges[:,1]], axis = 0), requires_grad = True)  

    # Here we prepare our constraints. First: The anglesum. For it we wish to have a list of which edges to check how. For further explanation, read documentation of Graphutil.getsurroundingedgelist()
    allsurroundings = gutil.getsurroundingedgelist(vertexnr, Graph.faces, edges.tolist())
    innersurrounds = [allsurroundings[i] for i in iv]
    
    for i in range(loops):
        # energy in this case is just how close we are to orthogonality:
        energy = torch.linalg.norm(N @ edgetensor) **2
        energies.append(energy.item())

        anglesum = optimizers.anglesum(innersurrounds, edgetensor)
        anglenergy = torch.linalg.norm((torch.zeros(anglesum.shape[0] + 2 * torch.pi) - anglesum))
        constraintenergies.append(anglenergy.item())

        fullenergy = energy + anglenergy
    
        fullenergy.backward()

        with torch.no_grad():
            edgetensor -= learnrate * edgetensor.grad # type: ignore

        torch.abs(edgetensor)

        # Reset Gradient
        edgetensor.grad.zero_() # type: ignore
    
    edgelengths = edgetensor.detach().numpy()
    # now we make the edgematrix that i've used for the last funcion cause it's easier for implementation (i think)
    edgematrix = np.zeros((vertexnr, vertexnr))
    for i in range(edgenr):
        edgematrix[edges[i,0], edges[i,1]] = edgematrix[edges[i,1], edges[i,0]] = edgelengths[i]
    
    print(edgematrix)

    newGraph = gutil.newreconstructfromedgelengths(list(Graph.faces), edgematrix)
    radii = gutil.spherepacker(Graph, edges, edgematrix)
    return newGraph, np.array(energies), radii






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
        AESgraph, energies, radii = optimizeviasvg(Graph, i * stepsize, learnrate = 0.001)
        print(radii)

        # axs[i,0].set_title("Soft conditions over optimization",fontsize = 7, y = -0.25)
        # print(outeredenergies[-1])

        axs[i,0].text(1.1,0.5, " Final AES\n  energy:\n     " + str(format(energies[-1], ".8f")), transform=axs[i,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)

        axs[i,0].set_title("AES minimized graph", fontsize = 7, y = -0.25)
        gutil.showGraph(AESgraph, axs[i,0])
        #gutil.visualizecircles(AESgraph.vertices, radii, axs[i,0])

        axs[i,1].set_title("AES energy over optimization",fontsize = 7, y = -0.25)
        axs[i,1].plot(energies)
        axs[i,1].text(1.1,0.5, " Final AES\n  energy:\n     " + str(format(energies[-1], ".8f")), transform=axs[i,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)


    
    plt.savefig(outputpath + "optimizeviasvg.pdf")

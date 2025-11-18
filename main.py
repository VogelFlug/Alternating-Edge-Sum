#Imports
import numpy as np
import torch
from matplotlib import pyplot as plt
import sys
from io import StringIO

#Internal Imports
from util.TwoDGraph import TwoDGraph
import util.Graphutil as gutil
import util.optimizationenergies as optimizers

#for testing purposes
import torchviz
import time



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
    # Buncha Graph stuff we need later
    vertices = Graph.vertices
    edges = gutil.getalledges(Graph.edgecounter)
    vertexnr = vertices.shape[1]
    edgenr = edges.shape[0]
    iv = np.array(gutil.ivfromscratch(vertexnr, Graph.edgecounter), dtype=int)
    nf = len(Graph.faces)
    Faces = torch.tensor(Graph.faces).flatten()

    # To produce some fun graphs to check out later
    energies = []
    constraintenergies = [[],[]]

    # The connectivity matrix has one row per edge and that row is entirely 0 except for the two vertices that make that edge
    connectivity = torch.zeros((edgenr,vertexnr))
    for i in range(edgenr):
        connectivity[i, edges[i,0]] = connectivity[i, edges[i,1]] = 1

    # get the pseudo inverse of the connectivity matrix to later get "fake" radii from current edgelength
    pseu_A = torch.linalg.pinv(connectivity)

    # Our problem can be broken down to connectivity @ radii = edges. Through singular value decomposition, we can find the core of the connectivity matrix, hidden in left singular vectors. 
    LS, Sig, RS = torch.linalg.svd(connectivity)
    N = LS[:,vertexnr:].T

    # And now we get to the actual optimization. Since N is orthogonal to some fitting edgelengths in the question of connectivity @ radii = edges, we wish to achieve exactly that:
    edgetensor = torch.tensor(np.linalg.norm(vertices[:,edges[:,0]] - vertices[:,edges[:,1]], axis = 0), requires_grad = True, dtype=torch.float32)  

    # Alright, new plan, time to make matrix magic happen: we will create three numpy arrays of size nf x 3, where for each face [i,j,k], we have the line [ij, jk, ik]
    faceedges = gutil.getedgefacelist(Graph.faces, edges.tolist())
    cyclefe = faceedges[:,[1,2,0]]
    fullcycfe = faceedges[:,[2,0,1]]

    optimalangles = torch.zeros(iv.shape[0]) + 2*torch.pi
    # Idk if this actually saves on computing time but eh:
    zeros = torch.zeros(vertexnr)
    anglefactor = 1

    print("start")
    for i in range(loops):
        #playing around a bit with the factors
        # if(edgetensor.isnan()[0] == True):
        #     print(i)
        # if(i == 25):
        #     anglefactor = 1
        #     learnrate *= 2500
        

        # energy in this case is just how close we are to orthogonality:
        energy = torch.linalg.norm(N @ edgetensor) ** 2
        energies.append(energy.item())

        # Get the energy representing how close we are to all innervertices having an anglesum of 360 degrees.
        # Each face [i,j,k] has 3 angles that need calculating, in the first column we calculate the angle at i, in the second column the angle at j and in the third the angle at k.
        # Thus each c array is built like [jk, ik, ij] (i.e. holding the edgelength of the opposite edge) and a and b are shifted the represent the other two edges

        c = edgetensor[faceedges]
        b = edgetensor[cyclefe]
        a = edgetensor[fullcycfe]

        # The angles are calculated via the law of consines
        a_ = torch.clamp((torch.square(b) + torch.square(a) - torch.square(c)) / 2 / b / a,min=-1+1e-6, max = 1 - 1e-6)
        angles = torch.arccos(a_)
        anglesums = torch.zeros((nf))
        # Check whether the anglesum around all inner vertices is 2pi
        anglesums.scatter_add_(0, Faces, angles.flatten())
        anglenergy = anglefactor * torch.linalg.norm(optimalangles - anglesums[iv])
        constraintenergies[0].append(anglenergy.item())

    



        # Punish negative radii (or at least the "simulated" radii) via the Pseudo inverse. The Pseudo inverse multiplied with the Edgelength gives us a set of "fake radii", 
        # we check if forbidding these from being negative prevents negative radii and thus nonense graphs
        pseu_rad = pseu_A @ edgetensor
        pseu_neg = torch.minimum(pseu_rad, zeros)
        radergy = torch.linalg.norm(pseu_neg)
        constraintenergies[1].append(radergy.item())

        fullenergy = energy + anglenergy + radergy
        fullenergy /= vertexnr

    
        fullenergy.backward()

        with torch.no_grad():
            #print(edgetensor.grad)
            edgetensor -= learnrate * edgetensor.grad # type: ignore

        if(i >1 and constraintenergies[0][-1] >= constraintenergies[0][-2]):
              learnrate/=1.00008
        torch.abs(edgetensor)

        # Reset Gradient
        edgetensor.grad.zero_() # type: ignore

    edgelengths = edgetensor.detach().numpy()
    # now we make the edgematrix that i've used for the last funcion cause it's easier for implementation (i think)
    edgematrix = np.zeros((vertexnr, vertexnr))
    for i in range(edgenr):
        edgematrix[edges[i,0], edges[i,1]] = edgematrix[edges[i,1], edges[i,0]] = edgelengths[i]
    
    newGraph = gutil.newreconstructfromedgelengths(list(Graph.faces), edgematrix)
    radii = gutil.spherepacker(Graph, edges, edgematrix)
    return newGraph, np.array(energies), radii, constraintenergies




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



    #create the plots, First row is for Original Graph and Tutte Embedding, every two rows after that are for one attempt, holding the attempted recreation of the graph, the minimized AES energy, and up to two constraintenergies
    fig, axs = plt.subplots(1 + 2*attempts,2)

    #plot input graph for reference
    axs[0,0].set_title("Original Graph", fontsize = 7)
    gutil.showGraph(Graph, axs[0,0])

    # plot the Tutte Embedding of the original Graph and add the AES value to the side (only 6 decimal points cause otherwise it will get very messy)
    axs[0,1].set_title("Tutte Embedding", fontsize = 7)
    TutteGraph = gutil.standardtuttembedding(Graph)
    gutil.showGraph(TutteGraph, axs[0,1])
    axs[0,1].text(1.1,0.5, "    AES energy\n  for Tutte: \n     " + str(format(optimizers.SnapshotAES(TutteGraph),".8f")), transform=axs[0,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)
    
    for i in range(1, 1 + attempts):
        AESgraph, energies, radii, constraintenergies = optimizeviasvg(Graph, i * stepsize, learnrate = 125)
        #print(radii)

        # axs[i,0].set_title("Soft conditions over optimization",fontsize = 7, y = -0.25)
        # print(outeredenergies[-1])

        axs[2*i-1,0].text(1.1,0.5, " Final AES\n  energy:\n     " + str(format(energies[-1], ".8f")), transform=axs[2*i-1,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)

        axs[2*i-1,0].set_title("AES minimized graph", fontsize = 7)
        gutil.showGraph(AESgraph, axs[i,0])
        gutil.visualizecircles(AESgraph.vertices, radii, axs[i,0])

        axs[2*i-1,1].set_title("AES energy over optimization",fontsize = 7)
        axs[2*i-1,1].plot(energies)
        axs[2*i-1,1].text(1.1,0.5, " Final AES\n  energy:\n     " + str(format(energies[-1], ".8f")), transform=axs[2*i-1,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)
        print( constraintenergies[0][-2],  constraintenergies[0][-1])

        # For the constraint energies, we assume we always implement two constraintenergies. If its less, the graphs remain empty, if its more...TODO
        axs[2*i,0].set_title("First Constraint Energy",fontsize = 7)
        axs[2*i,0].plot(constraintenergies[0])
        axs[2*i,1].set_title("Second Constraint Energy",fontsize = 7)
        axs[2*i,1].plot(constraintenergies[1])
        axs[2*i,1].text(1.1,0.5, " Final Total\n  energy:\n     " + str(format(energies[-1] + constraintenergies[0][-1], ".8f")), transform=axs[2*i,1].transAxes,  rotation = 0, va = "center", ha="center", fontsize=7)


    plt.tight_layout(pad = 0, w_pad=-20, h_pad=1)
    plt.savefig(outputpath + "optimizeviasvg.pdf")

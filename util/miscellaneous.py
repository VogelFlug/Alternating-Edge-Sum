#collection of functions that didn't fit anywhere else and aren't too important for us anymore

import numpy as np
import torch

from . import Graphutil as gutil
from .TwoDGraph import TwoDGraph
from .optimizationenergies import tutteenergy

#Tutteembedding via Gradient Descent
def gradienttutte(Graph: TwoDGraph, learnrate: float):
    vertices = Graph.vertices
    oe = gutil.getouteredges(Graph.edgecounter)
    ie = gutil.getinneredges(Graph.edgecounter)

    ov = gutil.getoutervertices(oe)
    iv = gutil.getinnervertices(Graph.vertexnumber, ov)

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
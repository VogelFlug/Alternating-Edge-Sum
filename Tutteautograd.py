#Imports
import numpy as np
import torch
import sys
from io import StringIO

#Internal Imports
from classes.TwoDGraph import TwoDGraph
import classes.Graphutil as util

def main(autopath: str):
    # Get datafile
    x = autopath
    print(x)
    data = ""
    with open(x , "r") as f:
        data = f.read()
    Graph = TwoDGraph(data)
    print(Graph.vertices)
    print(Graph.faces)
    print(Graph.neighbourhood)
    print(Graph.edgecounter)
    oe = util.getouteredges(Graph.edgecounter)
    ie = util.getinneredges(Graph.edgecounter)
    print(oe)
    print(ie)
    print(util.getoutervertices(oe))
    print(util.getinnervertices(ie))

if __name__ == '__main__':
    main(sys.argv[1])
#Imports
import numpy as np
import torch
import sys
from io import StringIO

#Internal Imports
from classes.TwoDGraph import TwoDGraph

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

if __name__ == '__main__':
    main(sys.argv[1])